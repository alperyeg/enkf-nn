# -*- coding: utf-8 -*-
#
# structural_plasticity.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

'''
This file is based on the Structural Plasticity example
of the nest simulator library

-----------------------
This example shows a simple network of two populations where structural
plasticity is used. The network's bulk has 100 neurons, 80% excitatory and
20% inhibitory. The simulation starts without any connectivity. A set of
homeostatic rules are defined, according to which structural plasticity will
be created


The network is build as follows:

                    SP
  ----       ----  -->  [OUTPUT, 20 ex, 4 in]
 | I  |     | B  | -->  [OUTPUT, 20 ex, 4 in]
 | N  | SP  | U  | -->  [OUTPUT, 20 ex, 4 in]
 | P  | --> | L  | -->  [OUTPUT, 20 ex, 4 in]
 | U  |     | K  | -->  [OUTPUT, 20 ex, 4 in]
 | T  |     |    | -->  [OUTPUT, 20 ex, 4 in]
 |    |     |    | -->  [OUTPUT, 20 ex, 4 in]
 |783 |     |80ex| -->  [OUTPUT, 20 ex, 4 in]
 |    |     |20in| -->  [OUTPUT, 20 ex, 4 in]
  ----       ----  -->  [OUTPUT, 20 ex, 4 in]
              /\
              |
              Poisson Gen

'''

import pickle

import growth_curves
import matplotlib.pyplot as pl
# IMPORT LIBS
import mnist.data
import mnist.spike_generator
import mnist.visualize
import nest
import numpy
import pandas as pd


class StructuralPlasticity:
    def __init__(self):
        # SIMULATION PARAMETERS
        self.input_type = 'bellec'
        # simulated time (ms)
        self.t_sim = 60. # 60000.0
        # simulation step (ms).
        self.dt = 0.1

        self.number_input_neurons = 80
        self.number_bulk_exc_neurons = 800
        self.number_bulk_inh_neurons = 200
        self.number_out_exc_neurons = 1
        self.number_out_inh_neurons = 1
        self.number_output_clusters = 10

        # Structural_plasticity properties
        self.update_interval = 100
        self.record_interval = 1000.0
        # rate of background Poisson input
        self.bg_rate = 10000.0
        self.neuron_model = 'iaf_psc_alpha'

        # SPECIFY NEURON PARAMETERS
        # self.model_params = {'tau_m': 10.0,  # membrane time constant (ms)
        #                     # excitatory synaptic time constant (ms)
        #                     'tau_syn_ex': 0.5,
        #                     # inhibitory synaptic time constant (ms)
        #                     'tau_syn_in': 0.5,
        #                     't_ref': 2.0,  # absolute refractory period (ms)
        #                     'E_L': -65.0,  # resting membrane potential (mV)
        #                     'V_th': -50.0,  # spike threshold (mV)
        #                     'C_m': 250.0,  # membrane capacitance (pF)
        #                     'V_reset': -65.0  # reset potential (mV)
        #                     }

        self.nodes_in = None
        self.nodes_bulk_e = None
        self.nodes_bulk_i = None
        self.nodes_out_e = None
        self.nodes_out_i = None

        self.mean_ca_e = []
        self.mean_ca_i = []
        self.total_connections_e = []
        self.total_connections_i = []

        self.mean_ca_e_out_0 = []
        self.mean_ca_i_out_0 = []
        self.total_connections_e_out_0 = []
        self.total_connections_i_out_0 = []

        self.psc_in = 585.0
        self.psc_e = 485.0
        self.psc_i = -485.0
        self.psc_c = 585.0
        self.psc_out = 100.0
        self.psc_ext = 6.2

        # ALL THE DIFFERENT GROWTH CURVES
        self.growth_curve_in_e = growth_curves.in_e

        self.growth_curve_bulk_e_e = growth_curves.bulk_e_e
        self.growth_curve_bulk_e_i = growth_curves.bulk_e_i
        self.growth_curve_bulk_i_e = growth_curves.bulk_i_e
        self.growth_curve_bulk_i_i = growth_curves.bulk_i_i

        self.growth_curve_out_e_e = growth_curves.out_e_e
        self.growth_curve_out_e_i = growth_curves.out_e_i
        self.growth_curve_out_i_e = growth_curves.out_i_e
        self.growth_curve_out_i_i = growth_curves.out_i_i

        # MNIST DATA HANDLING
        self.target_label = ['1']
        self.other_label = ['0', '2', '3', '4', '5', '6', '7', '8', '9']

        self.target_px = None
        self.target_lbl = None
        self.other_px = None
        self.other_lbl = None
        self.test_px = None
        self.test_lbl = None

    def get_mnist_data(self):
        self.target_px, self.target_lbl, self.test_px, self.test_lbl = \
            mnist.data.fetch(path='./mnist/mnist784_dat/',
                             labels=self.target_label)
        self.other_px, self.other_lbl, self.test_px, self.test_lbl = \
            mnist.data.fetch(path='./mnist/mnist784_dat/', \
                             labels=self.other_label)

    def prepare_simulation(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus({'resolution': self.dt,
                              'grng_seed': 0})

        nest.SetStructuralPlasticityStatus({
            'structural_plasticity_update_interval': self.update_interval,
        })
        self.get_mnist_data()

    def create_nodes(self):
        synaptic_elems_in = {
            'In_E_Axn': self.growth_curve_in_e,
        }
        synaptic_elems_bulk_e = {
            'Bulk_E_Den': self.growth_curve_bulk_e_e,
            'Bulk_I_Den': self.growth_curve_bulk_e_i,
            'Bulk_E_Axn': self.growth_curve_bulk_e_e,
        }
        synaptic_elems_bulk_i = {
            'Bulk_E_Den': self.growth_curve_bulk_i_e,
            'Bulk_I_Den': self.growth_curve_bulk_i_i,
            'Bulk_I_Axn': self.growth_curve_bulk_i_i,
        }

        self.nodes_in = nest.Create('iaf_psc_alpha',
                                    self.number_input_neurons,
                                    {'synaptic_elements': synaptic_elems_in})

        self.nodes_e = nest.Create('iaf_psc_alpha',
                                   self.number_bulk_exc_neurons,
                                   {
                                       'synaptic_elements': synaptic_elems_bulk_e})

        self.nodes_i = nest.Create('iaf_psc_alpha',
                                   self.number_bulk_inh_neurons,
                                   {
                                       'synaptic_elements': synaptic_elems_bulk_i})

        self.nodes_out_e = []
        self.nodes_out_i = []

        for ii in range(self.number_output_clusters):
            synaptic_elems_out_e = {
                'Out_E_Den_{}'.format(ii): self.growth_curve_out_e_e[ii],
                'Out_I_Den_{}'.format(ii): self.growth_curve_out_e_i[ii],
                'Out_E_Axn_{}'.format(ii): self.growth_curve_out_e_e[ii],
            }
            self.nodes_out_e.append(nest.Create('iaf_psc_alpha',
                                                self.number_out_exc_neurons,
                                                {
                                                    'synaptic_elements': synaptic_elems_out_e}))

            synaptic_elems_out_i = {
                'Out_E_Den_{}'.format(ii): self.growth_curve_out_e_i[ii],
                'Out_I_Den_{}'.format(ii): self.growth_curve_out_i_i[ii],
                'Out_I_Axn_{}'.format(ii): self.growth_curve_out_i_i[ii],
            }
            self.nodes_out_i.append(nest.Create('iaf_psc_alpha',
                                                self.number_out_inh_neurons,
                                                {
                                                    'synaptic_elements': synaptic_elems_out_i}))

    def create_synapses(self):
        nest.CopyModel('static_synapse', 'random_synapse')
        nest.SetDefaults('random_synapse',
                         {'weight': 1.,
                          'delay': 1.0})

    def create_input_spike_detectors(self):
        self.input_spike_detector = nest.Create("spike_detector",
                                                params={"withgid": True,
                                                        "withtime": True})
        nest.Connect(self.nodes_in, self.input_spike_detector)
        # nest.Connect(self.pixel_rate_generators, self.input_spike_detector)

    def get_external_input(self):
        self.train_px_one, self.train_lb_one, self.test_px_one, self.test_lb_one = \
            mnist.data.fetch(path='./mnist/mnist784_dat/', labels=['1'])
        self.train_px_other, self.train_lb_other, self.test_px_other, self.test_lb_other = \
            mnist.data.fetch(path='./mnist/mnist784_dat/',
                             labels=['0', '2', '3', '4', '5', '6', '7', '8',
                                     '9'])

    def set_external_input(self, iteration):
        random_id = numpy.random.randint(low=0, high=len(self.train_px_one))
        image = self.train_px_one[random_id]
        # Save image for reference
        plottable_image = numpy.reshape(image, (28, 28))
        pl.imshow(plottable_image, cmap='gray_r')
        pl.title('Index: {}'.format(random_id))
        pl.savefig('normal_input{}.eps'.format(iteration), format='eps')
        pl.close()
        if self.input_type == 'greyvalue':
            rates = mnist.spike_generator.greyvalue(image,
                                                    min_rate=1, max_rate=100)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        elif self.input_type == 'greyvalue_sequential':
            rates = mnist.spike_generator.greyvalue_sequential(image,
                                                               min_rate=1,
                                                               max_rate=100,
                                                               start_time=0,
                                                               end_time=783)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)

        else:
            train_spikes, train_spike_times = mnist.spike_generator.bellec_spikes(
               self.train_px_one[random_id], self.number_input_neurons, self.dt)
            for ii, ii_spike_gen in enumerate(self.pixel_rate_generators):
                iter_neuron_spike_times = numpy.multiply(train_spikes[:, ii],
                                                         train_spike_times)
                nest.SetStatus([ii_spike_gen],
                               {"spike_times": iter_neuron_spike_times[
                                   iter_neuron_spike_times != 0],
                                "spike_weights": [1500.] * len(
                                    iter_neuron_spike_times[
                                        iter_neuron_spike_times != 0])}
                               )

    def set_other_external_input(self, iteration):
        random_id = numpy.random.randint(low=0, high=len(self.train_px_other))
        image = self.train_px_other[random_id]
        # Save other image for reference
        plottable_image = numpy.reshape(image, (28, 28))
        pl.imshow(plottable_image, cmap='gray_r')
        pl.title('Index: {}'.format(random_id))
        pl.savefig('other_input{}.eps'.format(iteration), format='eps')
        pl.close()
        # rates = mnist.spike_generator.greyvalue(image,
        #                                         min_rate=1, max_rate=100)
        # generator_stats = [{'rate': w} for w in rates]
        # nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def test_external_input(self, iteration):
        random_id = numpy.random.randint(low=0, high=len(self.test_px_one))
        image = self.test_px_one[random_id]
        # Save image for reference
        plottable_image = numpy.reshape(image, (28, 28))
        pl.imshow(plottable_image, cmap='gray_r')
        pl.title('Test Index: {}'.format(random_id))
        pl.savefig('normal_input{}.eps'.format(iteration), format='eps')
        pl.close()
        # rates = mnist.spike_generator.greyvalue(image,
        #                                         min_rate=1, max_rate=100)
        # generator_stats = [{'rate': w} for w in rates]
        # nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def test_other_external_input(self, iteration):
        random_id = numpy.random.randint(low=0, high=len(self.test_px_other))
        image = self.test_px_other[random_id]
        # Save other image for reference
        plottable_image = numpy.reshape(image, (28, 28))
        pl.imshow(plottable_image, cmap='gray_r')
        pl.title('Test Index: {}'.format(random_id))
        pl.savefig('other_input{}.eps'.format(iteration), format='eps')
        pl.close()
        # rates = mnist.spike_generator.greyvalue(image,
        #                                         min_rate=1, max_rate=100)
        # generator_stats = [{'rate': w} for w in rates]
        # nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def connect_greyvalue_input(self, n_img):
        self.pixel_rate_generators = nest.Create("poisson_generator",
                                                 self.number_input_neurons)
        # Poisson to input neurons
        syn_dict = {"model": "random_synapse"}
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one",
                     syn_spec=syn_dict)
        # Input neurons to bulk
        syn_dict = {"model": "random_synapse", "weight": weights}
        print(self.nodes_e[0:len(self.nodes_in)])
        nest.Connect(self.nodes_in, self.nodes_e[0:len(self.nodes_in)],
                     "one_to_one", syn_spec=syn_dict)

    def connect_greyvalue_sequential_input(self, n_img):
        rates, starts, ends = mnist.spike_generator.greyvalue_sequential(
            self.target_px[n_img], start_time=0, end_time=783, min_rate=0,
            max_rate=10)
        # FIXME changed to len(rates) from len(offsets)
        self.pixel_rate_generators = nest.Create("poisson_generator", len(rates))
        # FIXME changed commented out
        # nest.SetStatus(pixel_rate_generators, generator_stats)
        # Poisson to input neurons
        syn_dict = {"model": "random_synapse"}
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one",
                     syn_spec=syn_dict)
        # Input neurons to bulk
        syn_dict = {"model": "random_synapse"}
        nest.Connect(self.nodes_in, self.nodes_e[0:len(self.nodes_in)],
                     "one_to_one", syn_spec=syn_dict)

    def connect_bellec_input(self):
        self.pixel_rate_generators = nest.Create("spike_generator",
                                                 self.number_input_neurons)
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one")
        weights = {'distribution': 'uniform',
                   'low': self.psc_i,  'high': self.psc_e,}
        syn_dict = {"model": "random_synapse", "weight": weights}
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.05 * self.number_bulk_exc_neurons)}
        nest.Connect(self.nodes_in, self.nodes_e,
                     conn_spec=conn_dict, syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.05 * self.number_bulk_inh_neurons)}
        nest.Connect(self.nodes_in, self.nodes_i,
                     conn_spec=conn_dict, syn_spec=syn_dict)

    # Set a very low rate to the input, for the case where no input is provided
    def clear_input(self):
        generator_stats = [{'rate': 1.0} for _ in
                           range(self.number_input_neurons)]
        nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def set_growthrate_output(self, output_region, input_on, iteration):
        for ii in range(self.number_output_clusters):
            if input_on:
                if ii == output_region:
                    gre = growth_curves.correct_input_growth_curve_e
                    gri = growth_curves.correct_input_growth_curve_i
                else:
                    gre = growth_curves.other_input_growth_curve
                    gri = growth_curves.other_input_growth_curve
            else:
                gre = growth_curves.no_input_growth_curve
                gri = growth_curves.no_input_growth_curve
            if iteration > 10:
                gre['growth_rate'] = gre['growth_rate'] / (iteration % 10)
                gri['growth_rate'] = gre['growth_rate'] / (iteration % 10)

            synaptic_elems_out_e = {
                'Out_E_Den_{}'.format(ii): gre,
                'Out_I_Den_{}'.format(ii): gri,
                'Out_E_Axn_{}'.format(ii): gre,
            }
            nest.SetStatus(self.nodes_out_e[ii], 'synaptic_elements_param',
                           synaptic_elems_out_e)

            synaptic_elems_out_i = {
                'Out_E_Den_{}'.format(ii): gre,
                'Out_I_Den_{}'.format(ii): gri,
                'Out_I_Axn_{}'.format(ii): gri,
            }
            nest.SetStatus(self.nodes_out_i[ii], 'synaptic_elements_param',
                           synaptic_elems_out_i)

    # After a couple of iterations we want to freeze the bulk. We will do this only by setting the 
    # growth rate to 0 in the dentritic synaptic elements to still allow new connections to 
    # the output population. 
    def freeze_bulk(self):
        freeze = {'growth_rate': 0.0}
        synaptic_elems_out_e = {
            'Bulk_E_Den': freeze,
            'Bulk_I_Den': freeze,
            # 'Bulk_E_Axn': freeze,
        }
        nest.SetStatus(self.nodes_e, 'synaptic_elements_param',
                       synaptic_elems_out_e)
        synaptic_elems_out_i = {
            'Bulk_E_Den': freeze,
            'Bulk_I_Den': freeze,
            # 'Bulk_I_Axn': freeze,
        }
        nest.SetStatus(self.nodes_i, 'synaptic_elements_param',
                       synaptic_elems_out_i)

    def connect_internal_bulk(self):
        # Connect bulk
        weights = {'distribution': 'uniform',
                   'low': 0.0,  'high': self.psc_e,}
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.09 * self.number_bulk_exc_neurons)}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_e, self.nodes_e, conn_dict, syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.1 * self.number_bulk_inh_neurons)}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_e, self.nodes_i, conn_dict, syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.12 * self.number_bulk_exc_neurons)}
        weights = {'distribution': 'uniform',
                   'low': self.psc_i,  'high': 0.0,}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_i, self.nodes_e, conn_dict, syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.08 * self.number_bulk_inh_neurons)}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_i, self.nodes_i, conn_dict, syn_spec=syn_dict)

    def connect_bulk_to_out(self):
        # Bulk to out
        weights = {'distribution': 'uniform',
                   'low': 0.0,  'high': self.psc_e,}
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.23 * self.number_output_clusters)}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_e, self.nodes_out_e[0], conn_dict,
                     syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.39 * self.number_output_clusters)}
        syn_dict = {"model": "random_synapse", "weight": weights}
        nest.Connect(self.nodes_e, self.nodes_out_i[0], conn_dict,
                     syn_spec=syn_dict)

    def connect_external_input(self, n_img):
        if self.input_type == 'bellec':
            self.connect_bellec_input()
        elif self.input_type == 'greyvalue':
            self.connect_greyvalue_input(n_img)
        elif self.input_type == 'greyvalue_sequential':
            self.connect_greyvalue_sequential_input(n_img)

    def record_ca(self):
        ca_e = nest.GetStatus(self.nodes_e, 'Ca'),  # Calcium concentration
        self.mean_ca_e.append(numpy.mean(ca_e))
        ca_i = nest.GetStatus(self.nodes_i, 'Ca'),  # Calcium concentration
        self.mean_ca_i.append(numpy.mean(ca_i))

        ca_e = nest.GetStatus(self.nodes_out_e[0],
                              'Ca'),  # Calcium concentration
        self.mean_ca_e_out_0.append(numpy.mean(ca_e))
        ca_i = nest.GetStatus(self.nodes_out_i[0],
                              'Ca'),  # Calcium concentration
        self.mean_ca_i_out_0.append(numpy.mean(ca_i))

    def clear_records(self):
        self.mean_ca_i_out_0.clear()
        self.mean_ca_e_out_0.clear()
        self.mean_ca_i.clear()
        self.mean_ca_e.clear()
        self.total_connections_e.clear()
        self.total_connections_i.clear()
        self.total_connections_e_out_0.clear()
        self.total_connections_i_out_0.clear()
        nest.SetStatus(self.input_spike_detector, {"n_events": 0})

    def record_connectivity(self):
        syn_elems_e = nest.GetStatus(self.nodes_e, 'synaptic_elements')
        syn_elems_i = nest.GetStatus(self.nodes_i, 'synaptic_elements')
        self.total_connections_e.append(sum(neuron['Bulk_E_Axn']['z_connected']
                                            for neuron in syn_elems_e))
        self.total_connections_i.append(sum(neuron['Bulk_I_Axn']['z_connected']
                                            for neuron in syn_elems_i))
        # Visualize the connections from output 0. Hard coded for the moment
        syn_elems_e = nest.GetStatus(self.nodes_out_e[0], 'synaptic_elements')
        syn_elems_i = nest.GetStatus(self.nodes_out_i[0], 'synaptic_elements')
        self.total_connections_e_out_0.append(
            sum(neuron['Out_E_Axn_0']['z_connected']
                for neuron in syn_elems_e))
        self.total_connections_i_out_0.append(
            sum(neuron['Out_I_Axn_0']['z_connected']
                for neuron in syn_elems_i))

    # Good to debug. The input is now working.
    def plot_input_spikes(self, id):
        spikes = nest.GetStatus(self.input_spike_detector, keys="events")[0]
        mnist.visualize.spike_plot(spikes, "Input spikes", id=id)

    def plot_data(self, id):
        fig, ax1 = pl.subplots()
        ax1.axhline(self.growth_curve_bulk_e_e['eps'],
                    linewidth=4.0, color='#FF9999')
        ax1.plot(self.mean_ca_e, 'r',
                 label='Ca Concentration Excitatory Neurons', linewidth=2.0)
        ax1.axhline(self.growth_curve_bulk_i_i['eps'],
                    linewidth=4.0, color='#9999FF')
        ax1.plot(self.mean_ca_i, 'b',
                 label='Ca Concentration Inhibitory Neurons', linewidth=2.0)
        # ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Time in [s]")
        ax1.set_ylabel("Ca concentration")
        ax2 = ax1.twinx()
        ax2.plot(self.total_connections_e, 'm',
                 label='Excitatory connections', linewidth=2.0, linestyle='--')
        ax2.plot(self.total_connections_i, 'k',
                 label='Inhibitory connections', linewidth=2.0, linestyle='--')
        # ax2.set_ylim([0, 2500])
        ax2.set_ylabel("Connections")
        # ax1.legend(loc=1)
        # ax2.legend(loc=4)
        pl.savefig('sp_mins_bulk_{}.eps'.format(id), format='eps')
        pl.close()

    # Plot the ca and conns from the output region 0. Fixed for the moment.
    def plot_data_out(self, id):
        fig, ax1 = pl.subplots()
        ax1.axhline(self.growth_curve_out_e_e[0]['eps'],
                    linewidth=4.0, color='#FF9999')
        ax1.plot(self.mean_ca_e_out_0, 'r',
                 label='Ca Concentration Excitatory Neurons', linewidth=2.0)
        ax1.axhline(self.growth_curve_out_i_i[0]['eps'],
                    linewidth=4.0, color='#9999FF')
        ax1.plot(self.mean_ca_i_out_0, 'b',
                 label='Ca Concentration Inhibitory Neurons', linewidth=2.0)
        # ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Time in [s]")
        ax1.set_ylabel("Ca concentration")
        ax2 = ax1.twinx()
        ax2.plot(self.total_connections_e_out_0, 'm',
                 label='Excitatory connections', linewidth=2.0, linestyle='--')
        ax2.plot(self.total_connections_i_out_0, 'k',
                 label='Inhibitory connections', linewidth=2.0, linestyle='--')
        # ax2.set_ylim([0, 2500])
        ax2.set_ylabel("Connections")
        # ax1.legend(loc=1)
        # ax2.legend(loc=4)
        pl.savefig('sp_mins_out_{}.eps'.format(id), format='eps')
        pl.close()

    def simulate(self):
        print("Starting simulation")
        sim_steps = numpy.arange(0, self.t_sim, self.record_interval)
        for i, step in enumerate(sim_steps):
            nest.Simulate(self.record_interval)
            if i % 20 == 0:
                print("Progress: " + str(i / 2) + "%")
            self.record_ca()
            self.record_connectivity()
        print("Simulation loop finished successfully")

    def checkpoint(self, id):
        # Input connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_in))
        f = open('conn_input_{}.bin'.format(id), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # Bulk connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_e))
        f = open('conn_bulke_{}.bin'.format(id), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_i))
        f = open('conn_bulki_{}.bin'.format(id), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # Out connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_e[0]))
        f = open('conn_oute_0_{}.bin'.format(id), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_i[0]))
        f = open('conn_outi_0_{}.bin'.format(id), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def save_connections():
    conn = nest.GetConnections()
    status = nest.GetStatus(conn)
    d = {'source': [], 'target': [], 'weight': []}
    for elem in status:
        d['source'].append(elem.get('source'))
        d['target'].append(elem.get('target'))
        d['weight'].append(elem.get('weight'))
    df = pd.DataFrame(d)
    df.to_pickle('./connections.pkl')
    df.to_csv('./connections.csv')


if __name__ == '__main__':
    numpy.random.seed(0)
    example = StructuralPlasticity()
    # Prepare simulation
    example.prepare_simulation()
    example.create_nodes()
    example.create_synapses()
    nest.PrintNetwork(depth=2)
    # nest.EnableStructuralPlasticity()
    example.connect_internal_bulk()
    example.connect_external_input(0)
    example.connect_bulk_to_out()
    # example.connect_internal_out()
    example.create_input_spike_detectors()
    example.get_external_input()
    # Start training 
    for i in range(1):
        print('Iteration {}'.format(i))
        example.clear_records()
        # Show a one
        example.set_external_input(i)
        # example.set_growthrate_output(0, True, i)
        example.simulate()
        save_connections()
        print("One was shown")
        import sys
        sys.exit()
        # No input
        # example.clear_input()
        # example.set_growthrate_output(0, False, i)
        example.simulate()
        print("No input was shown")
        # Show anything else
        example.set_other_external_input(i)
        # example.set_growthrate_output(1, True, i)
        example.simulate()
        print("Show something different")
        if i == 0:
            example.freeze_bulk()
        example.plot_input_spikes(i)
        example.plot_data(i)
        example.plot_data_out(i)
        example.checkpoint(i)

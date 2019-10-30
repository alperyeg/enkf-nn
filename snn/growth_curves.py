
#Growth between input neurons and the bulk
in_e = {
    'growth_curve': "gaussian",
    'growth_rate': 0.002,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
}

# Bulk exc neurons building exc synaptic elements
bulk_e_e = {
    'growth_curve': "gaussian",
    'growth_rate': 0.0002,  # (elements/ms)
    'continuous': False,
    'eta': -0.20,  # Ca2+
    'eps': 0.15,  # Ca2+
}
# Bulk exc neurons building inh synaptic elements
bulk_e_i = {
    'growth_curve': "gaussian",
    'growth_rate': 0.0002,  # (elements/ms)
    'continuous': False,
    'eta': -0.20,  # Ca2+
    'eps': 0.15,  # Ca2+
}
# Bulk inh neurons building exc synaptic elements
bulk_i_e = {
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': -0.10,  # Ca2+
    'eps': 0.10,  # Ca2+
}
# Bulk inh neurons building inh synaptic elements
bulk_i_i = {
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': -0.10,  # Ca2+
    'eps': 0.10,  # Ca2+
}

# Output exc neurons building exc synaptic elements
out_e_e = []
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.001,  # (elements/ms)
    'continuous': False,
    'eta': -0.2,  # Ca2+
    'eps': 0.2,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})

# Output exc neurons building inh synaptic elements
out_e_i = []
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.001,  # (elements/ms)
    'continuous': False,
    'eta': -0.2,  # Ca2+
    'eps': 0.2,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_e_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})

# Output inh neurons building exc synaptic elements
out_i_e = []
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': -0.0002,  # (elements/ms)
    'continuous': False,
    'eta':-0.1,  # Ca2+
    'eps': 0.1,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_e.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})

# Output inh neurons building inh synaptic elements
out_i_i = []
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': -0.0002,  # (elements/ms)
    'continuous': False,
    'eta':-0.1,  # Ca2+
    'eps': 0.1,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})
out_i_i.append({
    'growth_curve': "gaussian",
    'growth_rate': 0.0001,  # (elements/ms)
    'continuous': False,
    'eta': 0.0,  # Ca2+
    'eps': 0.05,  # Ca2+
})


no_input_growth_curve = {
    'growth_rate': 0.0001,  # (elements/ms)
    'eta': -0.005,  # Ca2+
    'eps': 0.005,  # Ca2+
}

other_input_growth_curve = {
    'growth_rate': 0.0001,  # (elements/ms)
    'eta': -0.1,  # Ca2+
    'eps': 0.1,  # Ca2+
}

correct_input_growth_curve_e = {
    'growth_rate': 0.0002,  # (elements/ms)
    'eta': -0.05,  # Ca2+
    'eps': 0.05,  # Ca2+
}

correct_input_growth_curve_i = {
    'growth_rate': -0.0001,  # (elements/ms)
    'eta': -0.025,  # Ca2+
    'eps': 0.025,  # Ca2+
}

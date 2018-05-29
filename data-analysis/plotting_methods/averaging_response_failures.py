#!usr/bin/env python

# exp name | ratio response failure | num trials
print(3840+6400+6400+19200)
print(6*(1120+1280+1280+1120+1120+1280))
l = [
("colour", 0.0145833333333333, 3840),
("contrast#", 0.01671875, 6400),
("noise", 0.01296875, 6400),
("eidolon", 0.0109895833333333, 19200),
("false-colour", 0.0292, 6*1120),
("lowpass", 0.0198, 6*1280),
("highpass", 0.0212, 6*1280),
("phase", 0.0293, 6*1120),
("power", 0.0251, 6*1120),
("rotation", 0.0225, 6*1280)
]

# loop over experiments
# calculate totals
total_num_trials = 0
total_num_fail2resp = 0
for exp in l:
    total_num_trials += exp[2]
    total_num_fail2resp += exp[1] * exp[2]
#assert total_num_trials == 82880, 'total_num_trials is ' + str(total_num_trials)

# calculate weigths
weights = []
for exp in l:
    weights.append(1.0 * exp[2] / total_num_trials)

# calculate weighted mean
ratios = [r[1] for r in l]
mean = sum([w*r for w, r in zip(weights, ratios)])

# calculate weighted std
div = sum([w*(r-mean)**2 for w,r in zip(weights, ratios)])
denom = sum(weights) * (len(weights)-1) / len(weights)
std = (div/denom)**0.5

print(mean)
print(std)

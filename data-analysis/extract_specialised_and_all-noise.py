import pandas as pd
import os

# for specialised networks
exps = ['colour-png', 'contrast-png', 'lowpass', 'highpass', 'phase-scrambling',
        'rotation', 'noise-png', 'salt-and-pepper-png']
nrs = ['26', '07', '08', '11', '16', '09', '03', '21']

for exp, nr in zip(exps,nrs):
    expname = exp + '-experiment'
    path = '../raw-data/fine-tuning/' + expname
    for fname in [csv for csv in os.listdir(path) if csv.endswith('.csv')]:
        if ('sixteen' + nr) in fname:
            df = pd.read_csv(path + '/' + fname)
            df['subj'] = 'specialised'
            fname_split = fname.split('_')
            session_csv = fname_split[-2] + '_' + fname_split[-1]
            df.to_csv(path + '/' + expname + '_specialised_' + session_csv,
                      index=False)


# for all-noise networks
for exp in exps:
    expname = exp + '-experiment'
    path = '../raw-data/fine-tuning/' + expname
    # get file names
    fnames18 = sorted([csv for csv in os.listdir(path) if 'sixteen18' in csv])
    fnames19 = sorted([csv for csv in os.listdir(path) if 'sixteen19' in csv])
    print(expname)
    
    if not expname in ['noise-png-experiment', 'salt-and-pepper-png-experiment']:
        # iterate over all existing sessions
        for f18, f19 in zip(fnames18, fnames19):
            df18 = pd.read_csv(path + '/' + f18)
            df19 = pd.read_csv(path + '/' + f19)
            df = pd.concat([df18, df19])
            df['subj'] = 'all-noise'
            fname_split = f18.split('_')
            session_csv = fname_split[-2] + '_' + fname_split[-1]
            df.to_csv(path + '/' + expname + '_all-noise_' + session_csv,
                      index=False)
            
    elif expname == 'noise-png-experiment':
        # iterate over all existing sessions
        for f18 in fnames18:
            df = pd.read_csv(path + '/' + f18)
            df['subj'] = 'all-noise'
            fname_split = f18.split('_')
            session_csv = fname_split[-2] + '_' + fname_split[-1]
            df.to_csv(path + '/' + expname + '_all-noise_' + session_csv,
                      index=False)
            
    elif expname == 'salt-and-pepper-png-experiment':
        # iterate over all existing sessions
        for f19 in fnames19:
            df = pd.read_csv(path + '/' + f19)
            df['subj'] = 'all-noise'
            fname_split = f19.split('_')
            session_csv = fname_split[-2] + '_' + fname_split[-1]
            df.to_csv(path + '/' + expname + '_all-noise_' + session_csv,
                      index=False)


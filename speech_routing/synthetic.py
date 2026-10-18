"""Analytic voiced-modulation and colored-noise task; not recorded speech."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


def waveform_pair(seed, frames=100, sample_rate=16000, hop_ms=10., snr_db=None):
    """Independent harmonic signal and colored-noise streams with known clean target."""
    hop=round(sample_rate*hop_ms/1000)
    if frames<2 or hop<1 or sample_rate<2000:
        raise ValueError('Use at least two hops and sample_rate>=2000')
    signal_seed,noise_seed,condition_seed=np.random.SeedSequence(seed).spawn(3)
    signal_rng=np.random.default_rng(signal_seed);noise_rng=np.random.default_rng(noise_seed)
    conditions=np.random.default_rng(condition_seed)
    time=np.arange(frames*hop)/sample_rate
    pitch=conditions.uniform(100,240)*(1+.025*np.sin(2*np.pi*conditions.uniform(2,4)*time))
    phase=2*np.pi*np.cumsum(pitch)/sample_rate
    envelope=.15+.85*np.sin(np.pi*conditions.uniform(2,4)*time+conditions.uniform(0,np.pi))**2
    signal=np.zeros_like(time)
    for harmonic in range(1,min(20,int(sample_rate/(2*240)))+1):
        frequency=harmonic*pitch.mean()
        formant=.15+np.exp(-((frequency-700)/400)**2)+.5*np.exp(-((frequency-1500)/600)**2)
        signal+=formant/harmonic*np.sin(harmonic*phase+signal_rng.uniform(-.15,.15))
    signal*=envelope
    signal*=conditions.uniform(.03,.2)/max(np.sqrt(np.mean(signal**2)),1e-12)
    rho=conditions.uniform(.2,.9)
    noise=lfilter([np.sqrt(1-rho*rho)],[1.,-rho],noise_rng.normal(size=len(time)))
    noise/=max(np.sqrt(np.mean(noise**2)),1e-12)
    snr=float(conditions.uniform(-5,15) if snr_db is None else snr_db)
    noise*=np.sqrt(np.mean(signal**2))*10**(-snr/20)
    return (signal+noise).reshape(frames,hop).astype('float32'),signal.reshape(frames,hop).astype('float32')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/voiced_modulation')
    parser.add_argument('--seed',type=int,default=13)
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--sample-rate',type=int,default=16000)
    parser.add_argument('--hop-ms',type=float,default=10.)
    for name,count in [('train',48),('validation',12),('calibration',12),('test',24)]:
        parser.add_argument('--'+name+'-utterances',type=int,default=count)
    args=parser.parse_args()
    counts={name:getattr(args,name+'_utterances') for name in ['train','validation','calibration','test']}
    if min(counts.values())<1 or counts['calibration']<9:
        parser.error('Use nonempty roles and at least nine independent calibration utterances')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    groups={};next_group=0
    for role_index,(role,count) in enumerate(counts.items()):
        pairs=[waveform_pair(np.random.SeedSequence([args.seed,role_index,index]).generate_state(1)[0],
                             args.frames,args.sample_rate,args.hop_ms) for index in range(count)]
        group=next_group+np.arange(count);next_group+=count
        groups[role]=group.tolist()
        np.savez_compressed(out/(role+'.npz'),x=np.stack([pair[0] for pair in pairs]),
                            y=np.stack([pair[1] for pair in pairs]),group=group,
                            source=np.array('analytic_voiced_modulation'))
    setting={'source':'analytic_voiced_modulation','seed':args.seed,'sample_rate':args.sample_rate,
             'hop_ms':args.hop_ms,'frames':args.frames,'groups':groups,
             'signal':'100-240Hz modulated harmonic source, fixed spectral formants, syllabic amplitude envelope',
             'noise':'independent AR(1) colored noise, rho=.2-.9, SNR=-5..15dB',
             'loudness_rms':[.03,.2],'measurements':'generated signals, not recorded or intelligible human speech'}
    (out/'setting.json').write_text(json.dumps(setting,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

import textgrid as tg
from pydub import AudioSegment
import os
import re
from banyan import SortedSet, OverlappingIntervalsUpdator # pip install banyan

def maximize_nonoverlapping_count(intervals):
    # build "interval" tree sorted by the end-point O(n log n)
    tree = SortedSet(intervals, key=lambda elem: (elem[1], (elem[1] - elem[0])),updator=OverlappingIntervalsUpdator)
    result = []
    while tree: # until there are intervals left to consider
        # pop the interval with the smallest end-point, keep it in the result
        result.append(tree.pop()) # O(log n)

        # remove intervals that overlap with the popped interval
        overlapping_intervals = tree.overlap(result[-1]) # O(m log n)
        tree -= overlapping_intervals # O(m log n)
    return result

id=1
punctuation = '&!！,;:?？"\'、，.。；'

if not os.path.exists('data/train'):
    os.mkdir('data/train')
if not os.path.exists('data/dev'):
    os.mkdir('data/dev')
if not os.path.exists('data/eval'):
    os.mkdir('data/eval')

tr_wav_scp = open("data/train/wav.scp", 'w+')
tr_text_scp = open('data/train/text', 'w+')
tr_utt2spk = open('data/train/utt2spk', 'w+')
tr_spk2utt = open('data/train/spk2utt', 'w+')

cv_wav_scp = open("data/dev/wav.scp", 'w+')
cv_text_scp = open('data/dev/text', 'w+')
cv_utt2spk = open('data/dev/utt2spk', 'w+')
cv_spk2utt = open('data/dev/spk2utt', 'w+')

eval_wav_scp = open("data/eval/wav.scp", 'w+')
eval_text_scp = open('data/eval/text', 'w+')
eval_utt2spk = open('data/eval/utt2spk', 'w+')
eval_spk2utt = open('data/eval/spk2utt', 'w+')

dataset='train'
for root,dirs,files in os.walk("/mnt/nas_data/AISHELL-4/aishell4_"+dataset+"/TextGrid"):
    for file in files:
        text_file = os.path.join(root,file)
        wav_file="/mnt/nas_data/AISHELL-4/aishell4_"+dataset+"/wav/"+text_file.split("/")[-1][:-9]+'.wav'
        tgrid = tg.TextGrid.fromFile(text_file)
        sound = AudioSegment.from_mp3(wav_file)

        intervals = []
        d = dict()

        for item in tgrid:
            for utt in item:
                if len(utt.mark.strip())>0:
                    intervals.append((utt.minTime,utt.maxTime))
                    d[(utt.minTime,utt.maxTime)]=utt.mark
        non_overlap_intervals = maximize_nonoverlapping_count(intervals)
        non_overlap_intervals = set(non_overlap_intervals)
        print("overlap ratio: ",1-len(non_overlap_intervals)/len(intervals))

        for item in tgrid:
            spk = item.name
            for utt in item:
                if len(utt.mark.strip())>0 and (utt.minTime,utt.maxTime) in non_overlap_intervals:
                    wav_name ='/mnt/nas_data/AISHELL-4/wav/'+dataset+'/'+str(id)+".wav"
                    start_time = float(utt.minTime)*1000
                    end_time = float(utt.maxTime)*1000
                    wav = sound[start_time:end_time]
                    wav.export(wav_name, format="wav")
                    
                    if id%20!=0:
                        tr_wav_scp.writelines(spk+'-'+str(id) +" "+ wav_name +" \n")
                        tr_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                        tr_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                        tr_text_scp.writelines(spk+'-'+str(id)+" "+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                    else:
                        cv_wav_scp.writelines(spk+'-'+str(id) +" "+ wav_name +" \n")
                        cv_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                        cv_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                        cv_text_scp.writelines(spk+'-'+str(id)+" "+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                                        
                    id += 1

dataset="eval"
for root,dirs,files in os.walk("/mnt/nas_data/AISHELL-4/aishell4_"+dataset+"/TextGrid"):
    for file in files:
        text_file = os.path.join(root,file)
        wav_file="/mnt/nas_data/AISHELL-4/aishell4_"+dataset+"/wav/"+text_file.split("/")[-1][:-9]+'.wav'
        tgrid = tg.TextGrid.fromFile(text_file)
        sound = AudioSegment.from_mp3(wav_file)

        intervals = []
        d = dict()

        for item in tgrid:
            for utt in item:
                if len(utt.mark.strip())>0:
                    intervals.append((utt.minTime,utt.maxTime))
                    d[(utt.minTime,utt.maxTime)]=utt.mark
        non_overlap_intervals = maximize_nonoverlapping_count(intervals)
        non_overlap_intervals = set(non_overlap_intervals)
        print("overlap ratio: ",1-len(non_overlap_intervals)/len(intervals))

        for item in tgrid:
            spk = item.name
            for utt in item:
                if len(utt.mark.strip())>0 and (utt.minTime,utt.maxTime) in non_overlap_intervals:
                    wav_name ='/mnt/nas_data/AISHELL-4/wav/'+dataset+'/'+str(id)+".wav"
                    start_time = float(utt.minTime)*1000
                    end_time = float(utt.maxTime)*1000
                    wav = sound[start_time:end_time]
                    wav.export(wav_name, format="wav")
                    
                    eval_wav_scp.writelines(spk+'-'+str(id) +" "+ wav_name +" \n")
                    eval_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                    eval_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                    eval_text_scp.writelines(spk+'-'+str(id)+" "+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                   
                    id += 1

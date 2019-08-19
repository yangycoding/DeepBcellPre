

def subSeqq(seq,id,num):    
    win = num
    subSeq = ''
    if (id-win)<0 and (id + win)>len(seq):
        for i in range(win-id):
            subSeq+='b'
        for i in range(0,len(seq)):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='b'
    elif (id-win)<0 and (id+win+1)<=len(seq):
        for i in range(win-id):
            subSeq+='b'
        for i in range(0,id+win+1):
            subSeq+=seq[i]
    elif (id-win)>=0 and (id+win+1) > len(seq):
        for i in range(id-win,len(seq)):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='b'
    elif (id-win)>=0 and (id+win+1) <= len(seq):
        for i in range(id-win,id+win+1):
            subSeq+=seq[i]
    return subSeq


def get_testval(fasta_file):   
    oneofkey_pos = []
    oneofkey_neg = []    
    with open(fasta_file,'r') as fp:
        win = 24       
        for line in fp:
            if not line.startswith('>'):
                line = line.strip('\n')
                for i in range(len(line)):
                    if (line[i].islower()):
                        subSeq = subSeqq(line,i,win)
                        subSeq = subSeq.upper()
                        final_seq = [0] + [AA for AA in subSeq]
                        oneofkey_neg.append(final_seq)
                        del subSeq,final_seq
                    else:
                        subSeq = subSeqq(line,i,win)
                        subSeq = subSeq.upper()
                        final_seq = [1] + [AA for AA in subSeq]
                        oneofkey_pos.append(final_seq)
                        del subSeq,final_seq
        return oneofkey_pos,oneofkey_neg
    
    
def get_train(fasta_file):   
    oneofkey_pos = []
    oneofkey_neg = []    
    with open(fasta_file,'r') as fp:
        win = 24       
        for line in fp:
            if not line.startswith('>'):
                line = line.strip('\n')
                for i in range(len(line)):
                    if (line[i].islower()):
                        subSeq = subSeqq(line,i,win)
                        if subSeq.islower():
                            subSeq = subSeq.upper()
                            final_seq = [0] + [AA for AA in subSeq]
                            oneofkey_neg.append(final_seq)
                            del final_seq
                        del subSeq
                    else:
                        subSeq = subSeqq(line,i,win)
                        subSeq = subSeq.upper()
                        final_seq = [1] + [AA for AA in subSeq]
                        oneofkey_pos.append(final_seq)
                        del subSeq,final_seq
        return oneofkey_pos,oneofkey_neg   

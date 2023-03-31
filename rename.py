

with open('filenames/driving_train.txt','r+') as f:
    #import pdb; pdb.set_trace()
    lines = f.readlines()
    print(lines)
    for line in lines:
        
        with open('filenames/driving_train_finalpass.txt','a') as f1:
            print(line)
            f1.write(line.replace('cleanpass','finalpass').replace('/TRAIN',''))



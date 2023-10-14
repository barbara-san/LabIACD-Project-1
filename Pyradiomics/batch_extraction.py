import os

#os.system('cmd /k "echo meow"')

patients = os.listdir()
#print(patients)
for p in patients:
    #print(p)
    if (not os.path.isdir(p)) or p == "out":
        continue
    #print(p)
    print("\n\nwe at " + p)
    folders1 = os.listdir(p)
    s = ""
    i = 0
    for f in folders1:
        folders2 = os.listdir(p + "\\" + f)
        if (len(folders2) > i):
            i = len(folders2)
            s = f
    # p/s
    pastas = os.listdir(p + "\\" + s)
    #print(pastas)
    main = ""
    for f in pastas:
        if ("NA" in f):
            main = f
        
    #print(main)
    b = 1;
    for f in pastas:
        if ("Segmentation" in f):
            os.system(f'cmd /c """python pyradiomics-dcm-b.py --input-image-dir {p}\{s}\{main} --input-seg-file "{p}\{s}\{f}\\1-1.dcm" --output-dir out\OutputSR --temp-dir out\TempDir --parameters Pyradiomics_Params.yaml --features-dict featuresDict.tsv --name {p}-{b}"""')
            b+=1
















import os

patients = os.listdir()

for patient in patients:
    
    if (not os.path.isdir(patient)) or patient == "out":
        continue
    
    folders1 = os.listdir(patient)
    path = ""
    content = 0
    
    for folder in folders1:
        folders2 = os.listdir(patient + "\\" + folder)
        if (len(folders2) > content):
            content = len(folders2)
            path = folder
    
    folders = os.listdir(patient + "\\" + path)
    main = ""
    
    for folder in folders:
        if not "Annotation" in folder:
            main = folder
            break
        
    seg_index = 1;
    
    for folder in folders:
        if "Segmentation" in folder:
            print("\n\nPACIENTE NUMERO " + patient + " - SEGMENTAÇÃO " + str(seg_index))
            os.system(f'cmd /c """python pyradiomics-dcm-b.py --input-image-dir "{patient}\{path}\{main}" --input-seg-file "{patient}\{path}\{folder}\\1-1.dcm" --output-dir out\OutputSR --temp-dir out\TempDir --parameters Pyradiomics_Params.yaml --features-dict featuresDict.tsv --name {patient}-{seg_index}"""')
            seg_index+=1

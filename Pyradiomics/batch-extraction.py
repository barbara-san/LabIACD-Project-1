import os

main_directory = os.listdir()

for patient in main_directory:
    # enter each patients folder, ignoring the folders for storing output and other files that may be inside the main directory
    if (not os.path.isdir(patient)) or patient == "out" or patient == "OutputSR" or patient == "TempDir":
        continue
    
    folders = os.listdir(patient) # list of folders inside patient (CT-scan and X-ray)
    path = ""
    content = 0
    
    # finding the CT-scan folder (has the biggest content - directories with each segmentation and directory with the input DICOM series)
    for folder in folders:
        segmentations_or_series = os.listdir(patient + "\\" + folder)
        if (len(segmentations_or_series) > content):
            content = len(segmentations_or_series)
            path = folder
    
    folders = os.listdir(patient + "\\" + path)
    main = ""

    # finding the series folder
    for folder in folders:
        if not "Annotation" in folder:
            main = folder
            break
        
    seg_index = 1
    
    # 
    for folder in folders:
        if "Segmentation" in folder:
            print("\n\nPACIENTE NUMERO " + patient + " - SEGMENTAÇÃO " + str(seg_index))
            os.system(f'cmd /c """python pyradiomics-dcm-b.py --input-image-dir "{patient}\{path}\{main}" --input-seg-file "{patient}\{path}\{folder}\\1-1.dcm" --parameters Pyradiomics_Params.yaml --features-dict featuresDict.tsv --name {patient}-{seg_index}"""')
            seg_index+=1

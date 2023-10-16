The Nodule model class holds the information from a single physicians annotation of a nodule >= 3mm class with a particular scan. A nodule has many contours, each of which refers to the contour drawn for nodule in each scan slice.

### Attributes

- **subtlety**: int, range = {1,2,3,4,5}
    
    Difficulty of detection. Higher values indicate easier detection.

    1. 'Extremely Subtle'
    2. 'Moderately Subtle'
    3. 'Fairly Subtle'
    4. 'Moderately Obvious'
    5. 'Obvious'

- **internalStructure**: int, range = {1,2,3,4}

    Internal composition of the nodule.

    1. 'Soft Tissue'
    2. 'Fluid'
    3. 'Fat'
    4. 'Air'

- **calcification**: int, range = {1,2,3,4,6}

    Pattern of calcification, if present.

    1. 'Popcorn'
    2. 'Laminated'
    3. 'Solid'
    4. 'Non-central'
    5. 'Central'
    6. 'Absent'

- **sphericity**: int, range = {1,2,3,4,5}

    The three-dimensional shape of the nodule in terms of its roundness.

    1. 'Linear'
    2. 'Ovoid/Linear'
    3. 'Ovoid'
    4. 'Ovoid/Round'
    5. 'Round'

- **margin**: int, range = {1,2,3,4,5}
    
    Description of how well-defined the nodule margin is.
    
    1. 'Poorly Defined'
    2. 'Near Poorly Defined'
    3. 'Medium Margin'
    4. 'Near Sharp'
    5. 'Sharp'

- **lobulation**: int, range = {1,2,3,4,5}
    
    The degree of lobulation ranging from none to marked
    
    1. 'No Lobulation'
    2. 'Nearly No Lobulation'
    3. 'Medium Lobulation'
    4. 'Near Marked Lobulation'
    5. 'Marked Lobulation'

- **spiculation**: int, range = {1,2,3,4,5}

    The extent of spiculation present.
    
    1. 'No Spiculation'
    2. 'Nearly No Spiculation'
    3. 'Medium Spiculation'
    4. 'Near Marked Spiculation'
    5. 'Marked Spiculation'

- **texture**: int, range = {1,2,3,4,5}

    Radiographic solidity: internal texture (solid, ground glass, or mixed).
    
    1. 'Non-Solid/GGO'
    2. 'Non-Solid/Mixed'
    3. 'Part Solid/Mixed'
    4. 'Solid/Mixed'
    5. 'Solid'

    **malignancy**: int, range = {1,2,3,4,5}
    
    Subjective assessment of the likelihood of malignancy, assuming the scan originated from a 60-year-old male smoker.
    
    1. 'Highly Unlikely'
    2. 'Moderately Unlikely'
    3. 'Indeterminate'
    4. 'Moderately Suspicious'
    5. 'Highly Suspicious'
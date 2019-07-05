def create(files):
    """
    Split all data on train and val
    :param files: list of paths to files with parsed data
    """

    # Combine all parsed files together
    labels = []
    for file in files:
        with open(file, 'r') as f:
            labels.extend([l.split() for l in f.read().split('\n') if len(l.split()) == 6])

    return labels
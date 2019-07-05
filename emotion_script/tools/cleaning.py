import os


def clear_training_data(clear_logs=False):
    """
    Clear training data if necessary
    :param clear_logs: to clear or not to clear logs
    """

    checkpoint_files = os.listdir('../checkpoints')
    for file in checkpoint_files:
        os.remove(os.path.join('../checkpoints', file))
    if clear_logs:
        logs = os.listdir('../logs')
        for log in logs:
            os.remove(os.path.join('../logs', log))

    print('Done!')


clear_training_data(clear_logs=True)

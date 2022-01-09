"""Main module."""

def read_text(path):
    """To read any text documents in txt files.

    Parameters
    ----------
    path : str
        Directory path to the txt file that you would like to read.

    Returns
    -------
    [type]
        [description]
    """
    with open(path, 'r', encoding='utf-8') as file:
        text = file.readlines()
        text = [word.replace('\n', '') for word in text]
        return text


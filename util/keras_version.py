from keras import __version__ as KVersion


def is_new_keras():
    print("version:", KVersion)
    if KVersion[0] == "1":
        return False
    if KVersion[0] == "3":
        return True
    version_val = KVersion.split(".")
    return int(version_val[1]) > 3

LAST_SAMELINE_LEN   = 0
LAST_SAMELINE       = False


def report(message, sameline=False):
    global LAST_SAMELINE_LEN
    global LAST_SAMELINE

    if sameline:
        print "\r         " + (' ' * LAST_SAMELINE_LEN) + "\r",
    elif LAST_SAMELINE:
        print "\n",

    print "[krotos] {}".format(message),

    if sameline:
        LAST_SAMELINE_LEN = len(message)
    else:
        print "\n",
        LAST_SAMELINE_LEN = 0

    LAST_SAMELINE = sameline

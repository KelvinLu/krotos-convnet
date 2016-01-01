def report_newline():
    print "\n",

def report(message, sameline=False):
    if sameline: print "\r",
    print "[krotos] {}".format(message),
    if not sameline: print "\n",

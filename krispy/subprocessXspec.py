import subprocess
import os

""" These functions/script is created to run an XSPEC session and pass certain XSPEC commands down the pipeline at certain times according to 
    the termal output during the XSPEC spectral fitting. I.e. trying to avoid having to manually input the final commands for the fitting.

    Comments: *"gohea" (or whatever you type to get "xspec" to work in the terminal) must be run first.
             *"writefits" must be used in the XSPEC batch script.
             *Fits and text (and log) files created in the XSPEC process will have the same name based on the fits name from "writefits" above. 
             *The "error" command also needs to be present in the XSPEC batch script too (this should be there anyway).
"""

def add2log(logFile, text):
    """A function to continuously add text to a log file.

    Parameters
    ----------
    logFile : str
        The log file to be created or appended to.
        
    text : str
        The text to be appended to the log file.
            
    Returns
    -------
    None.

    Example
    -------
    # to create a .log file with a first line and then add another line
    add2log("logFile.log", "Some text.")
    add2log("logFile.log", "Maybe some more text.")
    """
    # if the file string is and emtpy string just return it
    if len(logFile)==0:
        return

    with open(logFile, "a") as lf:
        lf.write(text)


def loc8fitsAndTxtName(xspecFile):
    """A function to find the name of the fits file you have defined in you XSPEC .xcm file.

    Parameters
    ----------
    xspecFile : str
        The XSPEC .xcm file that has the "writefits" command used in it.
            
    Returns
    -------
    The fits path+file name from the .xcm file.

    Example
    -------
    # to get fits path+file name from the .xcm file
    fitsFile = loc8fitsAndTxtName("apec1fit_fpm1_cstat.xcm")
    """
    with open(xspecFile, "r") as lf:
        for line in lf.readlines():
            if line.startswith("writefits"):
                # get the path and file name for the "writefits" line
                return line.split()[-1]


def maybeRemoveFile(file, remove=False):
    """Check if a file exists and remove it if you want.

    Parameters
    ----------
    file : str
        File you want to check the existence of and maybe remove.

    remove : bool
        To delete the file set to True. If Flase and the file exists an assertion error is thrown.
        Default: False
            
    Returns
    -------
    None.

    Example
    -------
    # to check if "./fitsFile.fits" exists and if it does remove it
    maybeRemoveFile("./fitsFile.fits", remove=True)
    """
    keep_going = True
    if os.path.isfile(file):
        if remove:
            remove_file = subprocess.run(["rm", file])
        else:
            keep_going = False
    assert keep_going, f"File {file} already exists and \'remove\' in CheckFitsAndTxtName() [or \'overwrite\' in runXSPEC()] is set to False. \nPlease remove to continue."


def runXSPEC(xspecBatchFile=None, logFile=False, overwrite=False):
    """Runs an XSPEC .xcm batch file and then, in the same XSPEC program, run other manual commands to complete 
    the XSPEC fitting process. This is because some of the final commands needed to complete the fitting process 
    in XSPEC needs to be run manually.

    Parameters
    ----------
    xspecBatchFile : str
        The batch XSPEC .xcm file. Must make use of the "writefits" command.
        Default: None

    logFile : bool
        Set to True and this creates a .log file that captures all of the output from teh fitting processes.
        The default is False but it is recommended to be set to True so that the resulting fit can be checked.
        The file name will be the same as the .fits file from xspecBatchFile.
        Default: False
            
    overwrite : bool
        In order to avoid files being overwritten (or in the case of the .fits file appended to) during the 
        fitting process. Setting this to True will check if a .fits, .txt. and .log file exist (using the 
        name defined with "writefits" fromxspecBatchFile) and delete them. If False with those files being
        present then an error will occur and the program will hault.
        Default: False

    Returns
    -------
    None

    Example
    -------
    # run a spectral fitting from the batch script "apec1fit_fpm1_cstat.xcm" while creating a .log file and 
                # deleting existing/conflicting files if they are there

    runXSPEC(xspecBatchFile="apec1fit_fpm1_cstat.xcm", logFile=True, overwrite=True)

    # if "apec1fit_fpm1_cstat.xcm" has "writefits folder/specFit.fits" then by the end you should have 3 files:
                # 1. folder/specFit.fits, 2. folder/specFit.txt, 3. folder/specFit.log
    """
    # "writefits" must be used in the batch xspec file, this line finds the file name used in "writefits"
    fitsFile = loc8fitsAndTxtName(xspecBatchFile)

    # check if a fits, text, and log file with the same name is there, if so remove them to continue or stop?
    maybeRemoveFile(fitsFile, remove=overwrite)
    txtFile = fitsFile[:-5]+".txt"
    maybeRemoveFile(txtFile, remove=overwrite)
    logFile = fitsFile[:-5]+".log" if logFile else ""
    maybeRemoveFile(logFile, remove=overwrite)
    

    # XSPEC commands to execute once opened a pipeline
    cmds = ["@"+xspecBatchFile, 
            "iplot ldata ufspec rat", 
            "wdata "+txtFile, 
            "exit", 
            "exit"]
    # need a newline at the end of each command
    commands = [c+"\n" for c in cmds]

    # open pipeline to terminal
    with subprocess.Popen(["xspec"],
                          stdin =subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True,
                          bufsize=0) as xspec:

        # Send xspec commands to stdin

        for number, command in enumerate(commands):
            # we have a pipeline open so send commands
            xspec.stdin.write(command)

            # since the commands are run inside another command (i.e. in xspec) subprocess doesn't recognise when they finish
            # therfore the program will hang if the command finishes and the "xspec.stdout.readline()" line is reached. Therfore
            # checks need to be in place so that once one command is finished we just move onto the next one (i.e. get out of 
            # the while loop before the code gets to the "xspec.stdout.readline()" line and hangs).

            # the "errors" xspec command is run in the batch script and is usually the last, also once the batch script is done 
            # the text " Current data and model not fit yet.\n" will appear. Once both of these have appeared then it should be 
            # OK to move on.
            errors_ran = False

            # the other commands at the moment give out a predictable number of lines without any prompting. In order for them 
            # all to get logged in the .log file (and for the program to not hang) then we keep track of the lines to know when 
            # to move onto the next manual command.
            lines4iplot = 0
            while True:

                line = xspec.stdout.readline()

                add2log(logFile, line)

                # print("Out: ", bytes(line.encode())) # for troubleshooting (byte strings show spaces and returns easier)


                if command==commands[0]:
                    if line.startswith("!XSPEC12>error"):
                        errors_ran = True 
                    if line==" Current data and model not fit yet.\n" and errors_ran==True:
                        add2log(logFile, f"\n***BATCH SCRIPT {cmds[number]} COMPLETE. NOW FOR MANUAL COMMANDS.***\n")
                        break
                    elif line.startswith("!XSPEC12>tclreadline::readline read {Continue error search in this direction? }\n"):
                        # sometimes this prompt occurs because of poor model-to-data fit, not enough data in range, etc. to get 
                        # a good fit quickly. Choose to let it keep trying by answering "y" each time as default for now.
                        xspec.stdin.write("y\n")

                elif command=="iplot ldata ufspec rat\n":
                    lines4iplot += 1
                    if lines4iplot==2:
                        add2log(logFile, f"\n***MANUAL COMMAND {cmds[number]} COMPLETE.***\n")
                        break

                elif command==commands[2]:
                    lines4iplot += 1
                    if lines4iplot==3:
                        add2log(logFile, f"\n***MANUAL COMMAND {cmds[number]} COMPLETE.***\n")
                        break

                elif command=="exit\n":
                    add2log(logFile, f"\n***MANUAL COMMAND {cmds[number]} COMPLETE.***\n")
                    break
        add2log(logFile, f"\n***FINISHED COMMANDS {cmds}.***")

    print("XSPEC batch script and manual commands run. Please check log file to ensure expected behaviour.")


def runXSPEC_customCommands(xspecBatchFile=None, logFile=False, overwrite=False, **kwargs):
    # a function to define your own "cmds" list and give conditions for when you should move on from 
    # each (e.g. number of lines to screen, after a particular line, etc.). "xspecBatchFile" will 
    # always be first.

    # will this be needed?
    pass


if __name__=="__main__":
    runXSPEC(xspecBatchFile="fit_apecbkn_2fpm_cstat.xcm", logFile=True, overwrite=True)

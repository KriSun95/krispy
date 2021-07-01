import subprocess
import os
import time

""" These functions/script is created to run an XSPEC session and pass certain XSPEC commands down the pipeline at certain times according to 
    the termal output during the XSPEC spectral fitting. I.e. trying to avoid having to manually input the final commands for the fitting.

    Comments:*source the HEA initialisation file ("gohea" for us (or whatever you type to get "xspec" to work in the terminal) must be run first).
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
    # if the file string is an emtpy string just return it
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


def locMcmcName(xspecFile):
    """A function to find the name of the mcmc fits file you have defined in you XSPEC .xcm file, 
    if it's there.

    Parameters
    ----------
    xspecFile : str
        The XSPEC .xcm file that has the "writefits" command used in it.
            
    Returns
    -------
    The mcmc fits path+file name from the .xcm file.

    Example
    -------
    # to get mcmc fits path+file name from the .xcm file
    fitsFile = loc8fitsAndTxtName("apec1fit_fpm1_cstat.xcm")
    """
    with open(xspecFile, "r") as lf:
        for line in lf.readlines():
            if line.startswith("chain run"):
                # get the path and mcmc file name for the "chain run" line
                return line.split()[-1]
    return ""


def locGainName(xspecFile):
    """A function to find the name of the gain fits file you have defined in you XSPEC .xcm file, 
    if it's there. (This is if you use my way of saving out a fits file with the gain result in 
    it from you .xcm file.)

    Parameters
    ----------
    xspecFile : str
        The XSPEC .xcm file that has the "writefits" command used in it.
            
    Returns
    -------
    The gain fits path+file name from the .xcm file.

    Example
    -------
    # to get gain fits path+file name from the .xcm file
    fitsFile = loc8fitsAndTxtName("apec1fit_fpm1_cstat.xcm")
    """
    with open(xspecFile, "r") as lf:
        for line in lf.readlines():
            if line.startswith("set finalFILE"):
                # get the path and mcmc file name for the "chain run" line
                return line.split()[-1]
    return ""


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


def runXSPEC(xspecBatchFile, logFile=False, overwrite=False):
    """Runs an XSPEC .xcm batch file and then, in the same XSPEC program, run other manual commands to complete 
    the XSPEC fitting process. This is because some of the final commands needed to complete the fitting process 
    in XSPEC needs to be run manually.

    Parameters
    ----------
    xspecBatchFile : str
        The batch XSPEC .xcm file. Must make use of the "writefits" command.

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

    # mcmc might have been run, check for this file name convention
    mcmcFile = locMcmcName(xspecBatchFile) # returns "" if this isn't there
    maybeRemoveFile(mcmcFile, remove=overwrite)
    # gain might have been allowed to vary and saved, check for this file name convention
    gainFile = locGainName(xspecBatchFile) # returns "" if this isn't there
    maybeRemoveFile(gainFile, remove=overwrite)
    

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
            outputLines = 0
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
                    elif line.startswith("!XSPEC12>tclreadline::readline read {Number of trials exceeded: continue fitting? }\n"):
                        # sometimes this prompt occurs because of really poor model-to-data fit, not enough data in range, etc. to get 
                        # a good fit quickly. Choose to let it keep trying by answering "n" each time as default for now as it can run 
                        # on for ages otherwise.
                        xspec.stdin.write("n\n")

                elif command=="iplot ldata ufspec rat\n":
                    outputLines += 1
                    if outputLines==2:
                        add2log(logFile, f"\n***MANUAL COMMAND {cmds[number]} COMPLETE.***\n")
                        outputLines = 0 # reset counter
                        break

                elif command==commands[2]:
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


def XSPECfit(directory, xspecBatchFile, logFile=False, overwrite=False, printTime=False):
    """Runs an XSPEC .xcm batch file(s) in the corresponding directory(s) and then, in the same XSPEC program, 
    run other manual commands to complete the XSPEC fitting process. This is because some of the final commands 
    needed to complete the fitting process in XSPEC needs to be run manually.

    Parameters
    ----------
    directory : str
        The the directory with the batch XSPEC .xcm file. Must make use of the "writefits" command. Could be a 
        list of .xcm files. Could be a list of directories. Be explicit, i.e. if the xcm file is in your current 
        directory use "./". The "directory" and "xspecBatchFile" parameters must have the same number of entries.

    xspecBatchFile : str
        The batch XSPEC .xcm file. Must make use of the "writefits" command. Could be a list of .xcm files. The 
        "directory" and "xspecBatchFile" parameters must have the same number of entries.

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

    printTime : bool
        Do you want the time (in seconds) that it took the fit to be printed.
        Default: False

    Returns
    -------
    None

    Example
    -------
    ## run a spectral fitting from the batch script "apec1fit_fpm1_cstat.xcm" while creating a .log file and 
                ## deleting existing/conflicting files if they are there

    XSPECfit(directory="./", xspecBatchFile="apec1fit_fpm1_cstat.xcm", logFile=True, overwrite=True)

    # if "apec1fit_fpm1_cstat.xcm" has "writefits folder/specFit.fits" then by the end you should have 3 files:
                # 1. folder/specFit.fits, 2. folder/specFit.txt, 3. folder/specFit.log

    ## run for multiple .xcm in different directories

    XSPECfit(directory=["dir1", "dir2"], xspecBatchFile=["f1.xcm", "f2.xcm"], logFile=True, overwrite=True)

    ## run for multiple .xcm in the same directories

    XSPECfit(directory=["dir1", "dir1"], xspecBatchFile=["f1.xcm", "f2.xcm"], logFile=True, overwrite=True)
    """

    # make them loops and so loop-able
    directory = directory if type(directory)==list else[directory]
    xspecBatchFile = xspecBatchFile if type(xspecBatchFile)==list else[xspecBatchFile]
    startingDir = os.getcwd()

    # change directory to each cxm file and run the fit then go back to the original directory to start again
    for d,xcm in zip(directory, xspecBatchFile):

        start = time.time()

        os.chdir(d)
        runXSPEC(xspecBatchFile=xcm, logFile=logFile, overwrite=logFile)
        os.chdir(startingDir)

        if printTime:
            print("Fit took: ", time.time()-start, " seconds")


if __name__=="__main__":
    test=False
    if test:
        runXSPEC(xspecBatchFile="fit_apecbkn_2fpm_cstat.xcm", logFile=True, overwrite=True)
    # print(lastLineInLog("./xspec/mod_apecbknp_2fpmab_cstat.log"))
    # print(lastLineInLog("./mylog"))

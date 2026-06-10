#!/usr/bin/env python3
"""This script contains helper classes that are called by the processor to perform various tasks for different jobs. Each stage must inhert from and implement this"""

import os
import subprocess


### for interacting with SLURM or not


def SLURM_spinupJob(command, settings_fname, flashName, jobNumber, maxJobs, numCPUs, slurm_settings):
    


    with open('./pipelineSlurmJob', 'w') as fout:

#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -c 1                    # number of cores; coupled to 8000 MB memory per core
#SBATCH -t 10:00                # maximum run time in [HH:MM:SS] or [MM:SS] or [minutes]
#SBATCH -p normal               # partition (queue); job can run up to 5 days

        fout.write("#!/bin/bash\n")
        fout.write( slurm_settings['run_time'] )
        fout.write("\n")
        fout.write("#SBATCH -c ")
        fout.write(str(numCPUs))
        fout.write("\n")

        fout.write( slurm_settings['run_time'] )
        fout.write("\n")
        fout.write( slurm_settings['partition'] )
        fout.write("\n")

        fout.write("#SBATCH --output=")
        fout.write(slurm_settings['_output_filePrefix'])
        fout.write("%j.log\n")

        for line in slurm_settings['extras']:
            fout.write(line)
            fout.write("\n")

        fout.write( command )
        fout.write( " " )
        fout.write( settings_fname )
        fout.write( " " )
        fout.write( flashName )
        fout.write( " " )
        fout.write( str(jobNumber) )
        fout.write( " " )
        fout.write( str(maxJobs) )
        fout.write( " " )
        fout.write( str(numCPUs) )
        fout.write( "\n" )

    subp = subprocess.run(['sbatch', './pipelineSlurmJob'], stdout=subprocess.PIPE)
    text = subp.stdout.decode()

    return int(text.split()[-1])

def SYSTEM_spinupJob(command, settings_fname, flashName, jobNumber, maxJobs, numCPUs, slurm_settings):
    
    command_list = command.split()

    command_list += [settings_fname, flashName, str(jobNumber), str(maxJobs), str(numCPUs)]
    p = subprocess.Popen(command_list)

    return p.pid




### now for the classes

class generic_jobInterface:
    """this is a generic interface that every other interface must implement"""

    def __init__(self, settings_fname, settings_data, flashOutputFolder, flashName, flashDatabase_row):
        self.settings_fname = settings_fname
        self.settings_data = settings_data
        self.flashName = flashName
        self.flashOutputFolder = flashOutputFolder
        self.flashDatabase_row = flashDatabase_row


    def stage_name(self):
        """return the name of this stage"""
        raise NotImplementedError

    ## Transitions "ready", -> "running"
    def launch_job(self, use_slurm, slurm_settings, num_processes, maxCPU_per_process ):
        """launch a job. If use_slurm is true than use slurm. If false than use subprocess. Is allowed to just run a short task.
        This function will return a string to store, which will be used to check if job is still running"""
        raise NotImplementedError

    ## Transitions "running" -> "complete",
    def job_is_running(self, state_string, current_process_IDs):
        """Return  number processes running, and True if job is still running, False if not.  Note, it is possible to return 0,True if job runs elsewhere"""
        raise NotImplementedError


    ## if both of these are turn than transitions   "complete" -> "ready" (of next stage)
    def all_processes_succsesful(self, state_string):
        """Check if processes completed sucsefully"""
        raise NotImplementedError

    def postProcessing(self):
        """perform any post-processing. Return True if succsesful, False otherwise"""
        raise NotImplementedError





class standard_jobInterface( generic_jobInterface ):
    """ this is the normal way of running jobs:
         No precurser taksk, runs a command in SLURM or system.
         Each job produces a COMPLETE_i where i is the job number
    """

    ## these must still be implemented"""
    def stage_name(self):
        """return the name of this stage"""
        raise NotImplementedError

    def command(self):
        """return string of command to run"""
        raise NotImplementedError


    ## these can be over-written to provide more complex behavior

    def initialize(self, use_slurm, slurm_settings, num_processes, maxCPU_per_process):
        return True


    def postProcessing(self):
        return True




    ## impletation meat

    def launch_job(self, use_slurm, slurm_settings, num_processes, maxCPU_per_process ):

        if not self.initialize(use_slurm, slurm_settings, num_processes, maxCPU_per_process):
            return None


        cmd = self.command()

        if use_slurm:
            spinupJob_function = SLURM_spinupJob
        else:
            spinupJob_function = getSystem_processIDs


        jobIDs = [ spinupJob_function(cmd, self.settings_fname, self.flashName, jobNumber, num_processes, maxCPU_per_process, slurm_settings) for jobNumber in range(num_processes)]

        return ':'.join( [str(j) for j in jobIDs] )


    def job_is_running(self, state_string, current_process_IDs):
        jobIDS = [int(i) for i in state_string.split(':')]

        num_running = 0
        for id in jobIDS:
            if id in current_process_IDs:
                num_running += 1

        return num_running, num_running>0

    def all_processes_succsesful(self, state_string):
        """Check if processes completed sucsefully"""

        jobIDS = [int(i) for i in state_string.split(':')]
        num_IDs = len(jobIDS)

        current_state = self.stage_name()

        completionFname = os.path.join(  self.flashOutputFolder, self.settings_data[current_state]['output_folderName'], 
                    self.settings_data[current_state]['log_folderName'], self.settings_data[current_state]['succesfulCompletion_FileName'] )


        have_all_fnames = True
        for i in range(num_IDs):
            fname = completionFname+"_"+str(i)
            if not os.path.isfile(fname):
                have_all_fnames = False
                break

        return have_all_fnames




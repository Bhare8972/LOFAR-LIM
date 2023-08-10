#!/usr/bin/env python3

from os import mkdir
from os.path import isdir


from LoLIM.utilities import logger, processed_data_dir
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1, read_cal_file
from LoLIM.findRFI import window_and_filter

from LoLIM.getTrace_fromLoc import getTrace_fromLoc 


from LoLIM.stationTimings.findEventbyLoc import save_EventByLoc
from LoLIM.stationTimings.timingInspector_4 import plot_all_stations



from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = 'D20190424T194432.504Z'
    output_folder = 'CalAttempt3_2023_pulses_even'

    totalCal_fname = 'TotalCal_6Jan2023.txt'


    initial_event_ID = 0
    event_XYZTs = [
    ## odd
    # [-41555.22938012603, -618.0196944709612, 9076.241351251778, 2.0830203178520392],
    # [-38302.058021527446, -4060.4919100103302, 9320.626039364053, 2.1781052222558306], 
    # [-38303.79380750979, -4059.8681937238475, 9323.89634054766, 2.1781057543061095], 
    # [-38068.99996942001, -5022.324687763768, 9775.552724041696, 2.1643159188584074],
    # [-32527.89143229711, -4548.661116315411, 6822.650122340187, 1.0688213031867175],
    # [-38418.4517721622, -4028.3890334879698, 9261.837819604876, 2.180098293550086], 
    # [-38303.75001097297, -3824.0433367526953, 8981.562312620354, 2.1864238116292487], 
    # [-37318.59306384959, -872.8371545326536, 8811.732905266384, 1.6994494058476632], 
    # [-37861.105047199184, -586.8709906846412, 9360.083245463924, 1.6637596901884795], 
    # [-41348.111326112856, -992.6416179051862, 8147.547855155281, 2.270669215254646],
    # [-37613.66632210442, -692.7859896076193, 9103.7862967621, 1.6870874542757974], 
    # [-37682.86237338734, -4676.602192456924, 9323.698458423203, 2.1752333311211953], 
    # [-41458.538819985035, -737.807598173779, 8910.8715502916, 2.0809880584961933], 
    # [-38542.43315150615, -5677.8726376275345, 10029.599633047279, 2.1576728803526333],
    # [-40503.05055142067, -1511.5294206459548, 8481.975621416641, 2.0572927985441263], 
    # [-40581.892865409296, -2935.3794383841905, 9148.537379693478, 2.1052693016672435], 
    # [-40697.98285546062, -1261.0664734722186, 8411.944848808223, 2.2668544174994003], 
    # [-37552.69622083565, -856.912701972239, 8938.315632660895, 1.695178524829914], 
    # [-30804.370082784128, -2731.9471714806227, 6907.66851072759, 2.2402740277024935],
    # [-40419.11028657052, -2416.070300740959, 8760.25582125396, 2.0290954861677344], 
    # [-37943.44014991651, -680.9452082216527, 9365.95127599061, 1.659554508732842],
    # [-37424.78126909382, -782.2672631607156, 8869.600426424242, 1.6955550082142465],
    # [-37611.67235266382, -639.8773565549317, 9027.540194589417, 1.6898579483224303], 
    # [-35353.815695626006, -2592.7896774088617, 6016.636475081612, 1.562772364351591],
    # [-40276.33246230777, -2336.5461328768824, 8461.907404376809, 2.033974368807652], 
    # [-40303.34363602675, -2306.1580148430253, 8470.473038665534, 2.034138760581282],
    # [-35702.15999868392, -2957.7739994647977, 5779.105756365274, 1.456867976965338], 

    ## even

[-38300.61046889991, -4142.896361470698, 9313.341279292747, 2.177563389307139], 
[-32083.468902843088, -5939.988159994213, 4254.654050024966, 2.0851774052402354],
[-32213.946568325857, -5086.847525475237, 4895.794298343571, 2.0832121993629324],
[-28896.892552497833, -9219.381380288629, 3120.755647251503, 1.5585032148411264], 
[-24434.616817856473, -9384.561064565562, 4844.995542352169, 1.6245449501681166], 
[-29135.07036386803, -10125.107134053176, 2767.5121107186524, 1.5539914521309832], 
[-24107.503006871244, -9782.7101288269, 4230.537058328213, 1.3069265612933159], 
[-28402.5324045491, -10223.922017197827, 2637.647003619955, 1.5467432166308395],
[-25851.249428494535, -11300.84905522105, 6878.706715574048, 1.5965206176990914], 
[-28777.75584099614, -5963.263385687198, 5035.668484716745, 2.3075690334172214], 
[-26894.078077739374, -11007.74878569042, 3126.8024896281445, 1.5301545733335187], 
[-28755.352037344703, -9353.458728780864, 2813.2012369514755, 1.553744665862793], 
[-28009.600813433037, -13928.405732200008, 8697.535489998863, 1.850111097047398], 
[-28487.4294413593, -11501.982677205955, 3001.2542530189694, 1.710200863634653],
[-27645.25086400428, -9658.291140093961, 3086.331818359139, 1.5438599913075342],
[-29573.27340285696, -10028.752370666492, 3528.6065159836767, 1.5727196739239353], 
[-27626.613000736823, -9828.583613123152, 3096.825219102497, 1.5428728433266057], 
[-32225.047407880145, -7110.240890231567, 2894.0655183473164, 2.1033536653623326], 
[-29249.87675681495, -8872.548928551523, 2822.642519145979, 1.5590695437109385], 
[-37690.64459110244, -4677.774647282174, 9322.888547191096, 2.1759536663674184], 
[-32225.51880823272, -6189.594753416245, 4245.738903061893, 2.08919533428352], 
[-33526.28235433752, -5472.629634632724, 4190.84389287183, 1.0519066065755986],
]


    first_event_todo = 0 ## index of above sources to start with. To skip things


    processed_data_folder = processed_data_dir(timeID)


    data_dir = processed_data_folder + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)

    totalCal_file = read_cal_file( processed_data_folder+'/'+totalCal_fname, pol_flips_are_bad=False )



    raw_fpaths = filePaths_by_stationName(timeID)

    print('opening files')

    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, total_cal=totalCal_file, pol_flips_are_bad=False) for sname,fpaths in raw_fpaths.items() if sname in totalCal_file.station_delays}

    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in totalCal_file.station_delays}

    trace_locator = getTrace_fromLoc( raw_data_files, data_filters )
    


    for ei, XYZT in enumerate(event_XYZTs):
        if ei < first_event_todo:
            continue

        event_ID = initial_event_ID+ei

        print('doing event' ,ei, '/', event_ID)

        ## this saves the pulse
        save_EventByLoc(event_ID, XYZT, data_dir, trace_locator, pulse_width=50, upsample_factor=2, min_ant_amp=2)
        print()
        print('at total event', initial_event_ID+ei)

        ## this plots it for you! 
        print('plot even')
        plot_all_stations(event_ID*10, 
               timeID, output_folder, pulse_input_folders=[output_folder], 
               guess_timings = totalCal_file.station_delays, 
               sources_to_fit=None,  ## not actually used! 
               guess_source_locations={ event_ID*10: XYZT },
               source_polarizations={ event_ID*10: 0 },       ## note this plots even 
               source_stations_to_exclude={ event_ID*10: [] }, source_antennas_to_exclude={ event_ID*10: [] }, bad_ants=[],
               antennas_to_recalibrate={}, min_ant_amplitude=2, ref_station="CS002", input_antenna_delays=None)


        print('plot odd')
        plot_all_stations(event_ID*10, 
               timeID, output_folder, pulse_input_folders=[output_folder], 
               guess_timings = totalCal_file.station_delays, 
               sources_to_fit=None,  ## not actually used! 
               guess_source_locations={ event_ID*10: XYZT }, 
               source_polarizations={ event_ID*10: 1 },       ## note this plots odd 
               source_stations_to_exclude={ event_ID*10: [] }, source_antennas_to_exclude={ event_ID*10: [] }, bad_ants=[],
               antennas_to_recalibrate={}, min_ant_amplitude=2, ref_station="CS002", input_antenna_delays=None)

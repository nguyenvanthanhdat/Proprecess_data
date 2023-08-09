from multiprocessing import Process

def print_func(continent='Asia'):
    print('The name of continent is : ', continent)

if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = Process(target=print_func)  # instantiating without any argument
    procs.append(proc) # save process
    proc.start() # run process

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=print_func, args=(name,)) # initial process
        procs.append(proc) # save process
        proc.start() # run process

    # complete the processes
    for proc in procs:
        proc.join() # shutdown process
    
    print(procs)
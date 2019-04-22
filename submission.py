import pandas as pd
import numpy as np
""" 
parses file into pandas array 
returns a pandas array
transition probability of row into column
NEED TO MAKE A METHOD TO SUPPORT BIG DATA
"""



def parse_state_file(file):
    file = open(file)
    lines = [line.split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]
    # data = [[1 for _ in range(col_count-1)] for _ in range(col_count-2)]
    # begin = np.ones(col_count)
    data = [[1 for _ in range(col_count-2)]+[0,1] \
             for _ in range(col_count-1)] + \
            [[0 for _ in range(col_count)]]
    
    
    
    for i in range(col_count+1, len(lines)):
        x,y,score = lines[i]
        data[int(x)][int(y)] = int(score) + 1
    for i in range(len(data)):
        total = sum(data[i])
        data[i] = list(map(lambda x:x/total if x!=0 else 0, data[i]))
    return cols, np.array(data)

def parse_symbol_file(file,state_cols):
    file = open(file)
    lines = [line.split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]
    out = np.ones((len(state_cols),col_count))
    for row in lines[(col_count+1):]:
        out[int(row[0])][int(row[1])] = int(row[2])+1
    for index in range(out.shape[0]):
        total = sum(out[index])
        prob = lambda t: t/(total+1)
        func = np.vectorize(prob)
        out[index]  = func(out[index])
    return cols, out

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    transProbs = parse_file(State_File,True)
    emsProbs = parse_file(Symbol_File, False)
    
    pass # Replace this line with your implementation...


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


def findVect(matrix, cols, symbls, value):
    try: 
        return matrix[:,symbls.index(value)]
    except ValueError:
        return np.log10([1/(len(cols)+len(symbls)+1)])



if __name__=="__main__":
    
    state_cols, state_matrix = parse_state_file('./dev_set/State_File')
    symbol_cols, symbol_matrix = parse_symbol_file("./dev_set/Symbol_File",state_cols)
    # y = np.array(['Unit5',',' ,'10', '-' ,'14' ,'Munt', 'St',',' ,'Bayswater',',' ,'WA', '6053'])

    # y = np.array(['RMB','526',',','Borden',',','WA','6338'])
    # y = np.array(['Dumbuoy', 'Rd',',' ,'Warracknabeal',',' ,'VIC' ,'3393'])
    # y = np.array(['MBF', '101a' ,'Pyke' ,'Rd',',', 'Mooroopna',',' ,'VIC' ,'3629'])
    # K = state_matrix.shape[0]
    # Pi = state_matrix[state_cols.index("BEGIN")]
    # T = len(y)
    # T1 = np.empty((K, T), 'd')
    # T2 = np.empty((K, T), 'B')
    # T1[:, 0] = Pi * findVect(symbol_matrix,state_cols,symbol_cols,y[0])
    # T2[:, 0] = state_cols.index("BEGIN")
    # for i in range(1, T):
    #     T1[:, i] = np.max(T1[:, i - 1] * state_matrix.T * findVect(symbol_matrix,state_cols,symbol_cols,y[i]).T, 1)
    #     T2[:, i] = np.argmax(T1[:, i - 1] * state_matrix.T, 1)
    # x = np.empty(T+1, 'B')
    # x[-1] = np.argmax(T1[:, T - 1])
    # for i in reversed(range(1,T)):
    #     x[i] = T2[x[i], i]
    
    with np.errstate(divide='ignore'):
        transition = np.log10(state_matrix)
        emission = np.log10(symbol_matrix)

    prev = []

    events = ['MBF' ,'101a','Pyke' ,'Rd',',', 'Mooroopna',',' ,'VIC' ,'3629']
    # events = ['RMB','526',',','Borden',',','WA','6338','END']
    
    # logprob = initial + findVect(emission,state_cols,symbol_cols,next(events))
    # emission[:, next(events)]
    p=logprob=transition[state_cols.index("BEGIN")] + findVect(emission,state_cols,symbol_cols,events[0])
    # prev.append([state_cols.index("BEGIN") for _ in range(transition.shape[0])])
    # print(np.argmin(p))
    for event in events:
        p = logprob[:, np.newaxis] + transition + findVect(emission,state_cols,symbol_cols,event)
        # if event=='MBF':
        #     print(p)
        logprob = np.max(p, axis=0)
        # if event != 'END'
        prev.append(np.argmax(p, axis=0))
    
    
    print(prev)
    best_state = np.argmax(logprob)
    state = best_state
    path = [state]
    for p in reversed(prev):
        state = p[state]
        path.append(state)
    
    print(path)
    # print(logprob(best_state]))
    #     print(i)
    #     x[i] = T2[x[i-1],i-1]
    
    # x[-2] = T2[x[-1],8]
    # x[-3] = T2[x[-2],7]
    # x[-4] = T2[x[-3],6]
    # x[-5] = T2[x[-4],5]
    # for i in reversed(range(2, T+1)):
    #     print(f"HELLO {T2[x[i-1]][i-1]} {x[i-1]} {i-1}")
        # x[i] = T2[x[i-1]][i-1]
    
    # T1[:, T-1] = np.max(T1[:, T-2]*state_matrix.T * findVect(symbol_matrix,state_cols,symbol_cols,y[T-1]).T, 1)
    # T1[:, T-1] = np.max(T1[:, T-1] * state_matrix.T * findVect(symbol_matrix,state_cols,symbol_cols,y[T]).T, 1)
    # T2[:, T+1] = np.argmax(T1[:, T+1 - 1] * state_matrix.T, 1)

    # x = np.empty(T, 'B')
    # x[-1] = np.argmax(T1[:, T - 1])
    # for i in reversed(range(1, T)):
    #     x[i - 1] = T2[x[i], i]
    # x=np.append([state_cols.index("BEGIN")], x)
    # for row in T1:
    #     print(row)
    # print(T1)
    # print(T2)
    # print(x)
    

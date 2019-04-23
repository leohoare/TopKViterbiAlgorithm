import pandas as pd
import numpy as np

def parse_state_file(file):
    file = open(file)
    lines = [line.split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]
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
    cols = [lines[i][0] for i in range(1, col_count+1)]+['UNK']
    out = np.ones((len(state_cols),col_count+1))
    # out[]
    for row in lines[(col_count+1):]:
        out[int(row[0]),int(row[1])] = int(row[2])+1
    for index in range(out.shape[0]):
        total = sum(out[index])
        prob = lambda t: t/(total)
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


def findVect(matrix,symbls, value):
    try: 
        return matrix[:,symbls.index(value)]
    except ValueError:
        return matrix[:,symbls.index('UNK')]

if __name__=="__main__":
    
    state_cols, state_matrix = parse_state_file('./dev_set/State_File')
    symbol_cols, symbol_matrix = parse_symbol_file("./dev_set/Symbol_File",state_cols)

    '''
    Unsure of where issues are arising, possibly around dealing with beg / end cases
    current method:
    Create transition from states
        N.N matrix
        End has 0 probability away from it
        Begin has 0 probability to it
        rest are calculated as ones and value overwritten if in state file (+1)
        then probabilities calculated from this
    create emission from symbols
        N.K matrix
        all initalised to zeros with column added for unknown
        values overwritten from symbol file (+1) 
        calculated probabilites
    1. find initial probabilities from the transition matrix (BEGIN state)
        please confirm?
    2. iterate through each observation in y 
        most of logic is here
        first line T1[:, i - 1] * state_matrix.T * findVect(symbol_matrix,symbol_cols,y[i]).T
            times last states * prob next state given state * emission prob of observed
            find max of this for each state // prob of that state
            to matrix running probabilities
        T2 -> this displays index of path. 
            similar
            last states probs * next states -> array of N by N, then for each N find max prob index.
    3. first find argmax of last state 

    '''

    # y = np.array(['Unit5',',' ,'10', '-' ,'14' ,'Munt', 'St',',' ,'Bayswater',',' ,'WA', '6053'])
    # y = np.array(['RMB','526',',','Borden',',','WA','6338'])
    # y = np.array(['Dumbuoy', 'Rd',',' ,'Warracknabeal',',' ,'VIC' ,'3393'])
    y = np.array(['BEGIN','MBF', '101a' ,'Pyke' ,'Rd',',', 'Mooroopna',',' ,'VIC' ,'3629','END'])
    
    K = state_matrix.shape[0]
    Pi = state_matrix[state_cols.index("BEGIN")]
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')
    # case for start case (force state to begin)
    T1[:, 0] = Pi 
    T2[:, 0] = state_cols.index('BEGIN')
    # iterate through all other cases
    for i in range(1, T-1):
        T1[:, i] = np.max(T1[:, i - 1] * state_matrix.T * findVect(symbol_matrix,symbol_cols,y[i]).T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * state_matrix.T, 1)
    # case for end case (force state to end)
    T1[:,T-1]=T1[:, i - 1] * state_matrix[:,state_cols.index('END')]
    T2[:,T-1]=state_cols.index('END')
    
    x = np.empty(T, 'B')
    x[-1] = 25
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i-1]
    print(x)
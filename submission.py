import pandas as pd
import numpy as np
import re


####  File Parsers ####

# parses state file
def parseStateFile(file):
    file = open(file)
    lines = [line.strip().split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]
    data = np.ones((col_count, col_count))
    data[:, cols.index('BEGIN')] = 0
    data[cols.index('END')] = 0
    for i in range(col_count+1, len(lines)):
        x,y,score = lines[i]
        data[int(x)][int(y)] = int(score) + 1
    for i in range(len(data)):
        total = sum(data[i])
        data[i] = list(map(lambda x:x/total if x!=0 else 0, data[i]))
    with np.errstate(divide='ignore'):
        return cols, np.log(data)
      
# parses symbol file  
def parseSymbolFile(file,N):
    file = open(file)
    lines = [line.strip().split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]+['UNK']
    out = np.ones((N,col_count+1))
    for row in lines[(col_count+1):]:
        out[int(row[0]),int(row[1])] = int(row[2])+1
    for index in range(out.shape[0]):
        total = sum(out[index])
        prob = lambda t: t/(total)
        func = np.vectorize(prob)
        out[index]  = func(out[index])
    with np.errstate(divide='ignore'):
        return cols, np.log(out)

# parses symbol file, with different probabilities for different kind
# of UNK symbols
def parseSymbolFile_advanced(file,N):
    file = open(file)
    lines = [line.strip().split() for line in file]
    col_count = int(lines[0][0])
    cols = [lines[i][0] for i in range(1, col_count+1)]+['UNK', 'UNK-N', 'UNK-T']
    out = np.ones((N,col_count+3))#p.concatenate((np.ones((18,col_count+3)),np.full((8,col_count+3),0.000000000001)))
    for row in lines[(col_count+1):]:
        out[int(row[0]),int(row[1])] = int(row[2])+1
    for index in range(out.shape[0]):
        total = sum(out[index])
        prob = lambda t: t/(total)
        func = np.vectorize(prob)
        out[index]  = func(out[index])
    
    K = 1/(44214*1000)
    out[:,-3][18:] = [K]*8
    out[:,-2][2:6] = K
    out[:,-2][7:9] = K
    out[:,-2][10:] = K
    out[:,-1][:2] = K
    out[:,-1][6] = K
    out[:,-1][9] = K
    out[:,-1][18:] = [K]*8
    out[:,cols.index(',')][:18] = [K]*18
    out[:,cols.index(',')][19:] = [K]*7
    out[:,cols.index('/')][:19] = [K]*19 
    out[:,cols.index('/')][20:] = [K]*6
    out[:,cols.index('-')][:20] = [K]*20 
    out[:,cols.index('-')][21:] = [K]*5
    out[:,cols.index('(')][:21] = [K]*21 
    out[:,cols.index('(')][22:] = [K]*4
    out[:,cols.index(')')][:22] = [K]*22 
    out[:,cols.index(')')][23:] = [K]*3
    out[:,cols.index('&')][:23] = [K]*23
    out[:,cols.index('&')][24:] = [K]*2  


    # out[:,cols.index('&')] = [0]*26  # I tried to replace 0 in all the elements except the one 
    # out[:,cols.index('&')][23] = 1   # with the right state, but this messed up the classification big time
    # out[:,cols.index(',')] = [0]*26 
    # out[:,cols.index(',')][18] = 1
    # out[:,cols.index('/')] = [0]*26 
    # out[:,cols.index('/')][19] = 1
    # out[:,cols.index('-')] = [0]*26 
    # out[:,cols.index('-')][20] = 1
    # out[:,cols.index('(')] = [0]*26 
    # out[:,cols.index('(')][21] = 1
    # out[:,cols.index('(')] = [0]*26 
    # out[:,cols.index(')')][22] = 1
    with np.errstate(divide='ignore'):
        return cols, np.log(out)

def parseQueryFile(file):
    file = open(file)
    return [parseAddress(line.strip()) for line in file]
    



####  Helpers ####

# helper to check if matrix contains value other returns unknown probabilities
def findVect(matrix,symbls, value):
    try:
        return matrix[:,symbls.index(value)]
    except ValueError:
        return matrix[:,symbls.index('UNK')]

def findVect2(matrix,symbls, value):
    try: 
        return matrix[np.newaxis, :,symbls.index(value)]
    except ValueError:
        return matrix[np.newaxis, :,symbls.index('UNK')]

def findValue(matrix,symbls, x, y):
    try: 
        return matrix[x,symbls.index(y)]
    except ValueError:
        return matrix[x,symbls.index('UNK')]

#Helpers for advanced decoding
def findVect_adv(matrix,symbls, value):
    try: 
        return matrix[:,symbls.index(value)]
    except ValueError:
        return return_unk(matrix,symbls, value)

def return_unk(matrix, symbls, value):
    if value.isdigit():
        return matrix[:,symbls.index('UNK-N')]
    elif bool(re.match('^[A-Za-z]+$', value)):
        return matrix[:,symbls.index('UNK-T')]
    else:
        return matrix[:,symbls.index('UNK')]

def findVect_adv2(matrix,symbls, value):
    try: 
        return matrix[np.newaxis, :,symbls.index(value)]
    except ValueError:
        return return_unk2(matrix, symbls, value)

def return_unk2(matrix, symbls, value):
        
    if value.isdigit():
        return matrix[np.newaxis, :,symbls.index('UNK-N')]
    if bool(re.match('^[A-Za-z]+$', value)):
        return matrix[np.newaxis, :,symbls.index('UNK-T')]
    return matrix[np.newaxis, :,symbls.index('UNK')]

def findValue(matrix,symbls, x, y):
    try: 
        return matrix[x,symbls.index(y)]
    except ValueError:
        return matrix[x,symbls.index('UNK')]



#helper to find address
def parseAddress(string):
    out = []
    running = ''
    for ch in string:
        if ch in [',','(',')','/','&','-','&', ' ']:
            if running:
                out.append(running)
                running=''
            if ch != ' ':
                out.append(ch)
        else:
            running += ch
    if running:
        out.append(running)
    out.append('END')
    return out



# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    state_cols, state_matrix = parseStateFile(State_File)
    symbol_cols, symbol_matrix = parseSymbolFile(Symbol_File,len(state_cols))
    queries = parseQueryFile(Query_File)
    out = []
    for query in queries:
        N = len(state_cols)
        Q = len(query)
        logprobs    = np.empty((N,Q), 'd')
        paths       = np.empty((N,Q), 'B')
        # special case for begin
        logprobs[:, 0] = state_matrix[state_cols.index("BEGIN")] +  findVect(symbol_matrix,symbol_cols,query[0])
        paths[:, 0] = state_cols.index("BEGIN")
        # normal cases
        for i in range(1, Q):
            logprobs[:, i] = np.max(logprobs[:, i - 1] + state_matrix.T + findVect2(symbol_matrix,symbol_cols,query[i]).T, 1)
            paths[:, i] = np.argmax(logprobs[:, i - 1] + state_matrix.T, 1)
        # case for end
        logprobs[:,Q-1] = np.max(logprobs[:, Q-2] + state_matrix.T)
        # build path
        path = [0 for _ in range(Q)]
        path[-1] = state_cols.index("END")
        for i in reversed(range(1, Q)):
            path[i - 1] = paths[path[i], i]
        path = [state_cols.index("BEGIN")] + path + [np.max(logprobs[:,Q-1])]
        # print(paths)
        out.append(path)
    return out

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    state_cols, state_matrix = parseStateFile(State_File)
    symbol_cols, symbol_matrix = parseSymbolFile(Symbol_File,len(state_cols))
    queries = parseQueryFile(Query_File)
    out = []
    for query in queries:
        N = len(state_cols)
        Q = len(query)
        logprobs    = np.empty((N,Q,k), 'd')
        paths       = np.empty((N,Q,k), 'B')
        logprobs[:,0,0] = state_matrix[state_cols.index("BEGIN")] + findVect(symbol_matrix,symbol_cols,query[0])
        for i in range(1,k):
            logprobs[:,0,i] = -1000
        paths[:, 0] = state_cols.index("BEGIN")
        for q in range(1, Q):
            for x in range(N):
                queue = []    
                for y in range(N):
                    for i in range(k):
                        prob = logprobs[y,q-1,i] + state_matrix[y,x] + findValue(symbol_matrix,symbol_cols,x,query[q])
                        queue.append((prob,y))
                queue.sort(key=lambda x: x[0], reverse=True)                
                for i in range(k):
                    logprobs[x,q,i] = queue[i][0]
                    paths[x,q,i]    = queue[i][1]
        results = []
        for K in range(k):
            logprobs[:,Q-1,K] = np.max(logprobs[:, Q-2,K] + state_matrix.T)
            path = [0 for _ in range(Q)]
            path[-1] = state_cols.index("END")
            for i in reversed(range(1, Q)):
                path[i - 1] = paths[path[i], i,K]
            path = [state_cols.index("BEGIN")] + path + [np.max(logprobs[:,Q-1,K])]
            results.append(path)
        # print(results)
        out.append(results)
    return out


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    state_cols, state_matrix = parseStateFile(State_File)
    symbol_cols, symbol_matrix = parseSymbolFile_advanced(Symbol_File,len(state_cols))
    queries = parseQueryFile(Query_File)
    out = []
    ppp = True
    for query in queries:
        N = len(state_cols)
        Q = len(query)
        logprobs    = np.empty((N,Q), 'd')
        paths       = np.empty((N,Q), 'B')
        # special case for begin
        logprobs[:, 0] = state_matrix[state_cols.index("BEGIN")] +  findVect_adv(symbol_matrix,symbol_cols,query[0])
        paths[:, 0] = state_cols.index("BEGIN")
        # normal cases
        for i in range(1, Q):
            logprobs[:, i] = np.max(logprobs[:, i - 1] + state_matrix.T + findVect_adv2(symbol_matrix,symbol_cols,query[i]).T, 1)
            paths[:, i] = np.argmax(logprobs[:, i - 1] + state_matrix.T, 1)
        # case for end
        logprobs[:,Q-1] = np.max(logprobs[:, Q-2] + state_matrix.T)
        # build path
        path = [0 for _ in range(Q)]
        path[-1] = state_cols.index("END")
        for i in reversed(range(1, Q)):
            path[i - 1] = paths[path[i], i]
        path = [state_cols.index("BEGIN")] + path + [np.max(logprobs[:,Q-1])]
        out.append(path)

    return out


# if __name__=="__main__":
    # pass
    # print(parseAddress('P.O Box 6196, St.Kilda Rd Central, Melbourne, VIC 3001'))
    # print(advanced_decoding('./dev_set/State_File','./dev_set/Symbol_File','./dev_set/Query_File'))
    # print(viterbi_algorithm('./dev_set/State_File','./dev_set/Symbol_File','./dev_set/Query_File'))
    # print(top_k_viterbi('./dev_set/State_File','./dev_set/Symbol_File','./dev_set/Query_File',2))
    


    # '''
    # Unsure of where issues are arising, possibly around dealing with beg / end cases
    # current method:
    # Create transition from states
    #     N.N matrix
    #     End has 0 probability away from it
    #     Begin has 0 probability to it
    #     rest are calculated as ones and value overwritten if in state file (+1)
    #     then probabilities calculated from this
    # create emission from symbols
    #     N.K matrix
    #     all initalised to zeros with column added for unknown
    #     values overwritten from symbol file (+1) 
    #     calculated probabilites
    # 1. find initial probabilities from the transition matrix (BEGIN state)
    #     please confirm?
    # 2. iterate through each observation in y 
    #     most of logic is here
    #     first line T1[:, i - 1] * state_matrix.T * findVect_adv(symbol_matrix,symbol_cols,y[i]).T
    #         times last states * prob next state given state * emission prob of observed
    #         find max of this for each state // prob of that state
    #         to matrix running probabilities
    #     T2 -> this displays index of path. 
    #         similar
    #         last states probs * next states -> array of N by N, then for each N find max prob index.
    # 3. first find argmax of last state 


"""     print( out[:,cols.index(',')],\
        out[:,cols.index('/')],\
            out[:,cols.index('-')],\
                out[:,cols.index('(')], \
                    out[:,cols.index(')')], \
                        out[:,cols.index('&')], sep = '\n|||||\n') 
    out[:,cols.index('&')][23] = 1  # this seems to work pretty well but still sometimes produces wrong labels 
    out[:,cols.index(',')][18] = 1
    out[:,cols.index('/')][19] = 1
    out[:,cols.index('-')][20] = 1  # including only up to this line gives 122 wrong labels
    out[:,cols.index('(')][21] = 1  # adding this last two lines gives 118 wrong labels
    out[:,cols.index(')')][22] = 1  # but the sentences that contain ( and ) are all classified with all 0's
"""

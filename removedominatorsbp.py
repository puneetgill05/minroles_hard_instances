#! /usr/bin/python3
import ast
import time
import datetime
import sys
import os
import re
import subprocess
import time
#import networkx as nx
#from pyvis.network import Network

# sys.path.append('/minedgerolemining')

from readup import readup, uptopu

def hasbeenremoved(e, em):
    if e in em:
        return True
    else:
        return False

def isneighbour(e, f, up):
    # is f a neighbour of e?
    #if ((e[1] - len(up)) in up[f[0]]) and ((f[1] - len(up)) in up[e[0]]):
    if ((e[1]) in up[f[0]]) and ((f[1]) in up[e[0]]):
        return True
    else:
        return False

def neighbours(e, em, up, pu):
    s = set()
    u = e[0]
    p = e[1]

    # every <u', p> is a neighbour
    uprimes = set()
    for uprime in pu[p]:
        if uprime == u:
            continue
        uprimes.add(uprime)
        f = tuple((uprime, p))
        if not hasbeenremoved(f, em):
            s.add(f)

    # every <u, p'> is a neighbour
    pprimes = set()
    for pprime in up[u]:
        if pprime == p:
            continue
        pprimes.add(pprime)
        f = tuple((u, pprime))
        if not hasbeenremoved(f, em):
            s.add(f)

    # for every uprime in uprimes, pprime in pprimes,
    # <uprime,pprime> is a potential neighbour
    for uprime in uprimes:
        for pprime in pprimes:
            if pprime not in up[uprime]:
                continue
            f = tuple((uprime,pprime))
            if (not hasbeenremoved(f, em)) and isneighbour(e, f, up):
                s.add(f)

    return s

def removedominators(em, dm, up, seq):
    pu = uptopu(up)
    fixpoint = False

    fpiter = 1
    while not fixpoint:
        print('removedominators, fixpoint iteration #', fpiter)
        sys.stdout.flush()
        fpiter += 1

        fixpoint = True
        edgenum = 1
        for u in up:
            for p in up[u]:
                if not (edgenum % 1000):
                    print('removedominators, edgenum:', edgenum)
                    sys.stdout.flush()
                edgenum += 1

                e = tuple((u, p))

                if hasbeenremoved(e, em):
                    continue

                newseq = removedominatorsonce(e, em, dm, up, pu, seq)

                if newseq != seq:
                    fixpoint = False

                seq = newseq
    return seq

def removedominatorsonce(e, em, dm, up, pu, seq):
    ne = neighbours(e, em, up, pu)
    necopy = ne.copy()
    for f in necopy:
        # Are all neighbours of e also neighbours of f?
        isdom = True
        for n in ne:
            if n == f:
                continue
            if not isneighbour(n, f, up):
                isdom = False
                break
        if isdom:
            # f is a dominator
            em[f] = tuple((e[0], e[1], seq))
            t = tuple((e[0], e[1]))
            if t not in dm:
                dm[t] = set()
            (dm[t]).add(f)
            seq += 1
            ne.remove(f)

    if not ne:
        # e is a zero-neighbour edge
        em[e] = tuple((-1, -1, seq))
        t = tuple((-1, -1))
        if t not in dm:
            dm[t] = set()
        (dm[t]).add(e)
        seq += 1

    return seq

def saveem(em, filename):
    f = open(filename, 'w')
    for e in em:
        f.write(str(e)+':'+str(em[e])+'\n')
    f.close()


def get_em(upfilename: str):
    up = readup(upfilename)
    emfilename = upfilename + '-em.txt'
    pu = uptopu(up)

    if os.path.isfile(emfilename):
        em = readem(emfilename)
    else:
        print('Removing dominators...')
        sys.stdout.flush()

        timeone = time.time()
        em = dict()
        dm = dict()
        seq = 0
        seq = removedominators(em, dm, up, seq)
        timetwo = time.time()

        print('done! Time taken:', timetwo - timeone)
        sys.stdout.flush()

        nedges = 0
        for u in up:
            nedges += len(up[u])

        print("Original # edges:", nedges)
        print('# dominators removed:', seq)
       # print('edge-marks:')
       # for e in em:
       #    print('\t'+str(e)+': ' + str(em[e]))

        print('Saving em to', emfilename, end='...')
        sys.stdout.flush()
        saveem(em, emfilename)
    return em


def readem(filename):
    em = dict()
    f = open(filename, 'r')
    for line in f:
        # r = re.findall(r'-?\d+\.?\d*', line)
        # r = re.findall(r"'[^']*'|\d+", line)
        r = line.split(':')
        e = ast.literal_eval(r[0].strip())
        m = ast.literal_eval(r[1].strip())
        # m = (m[0], m[1], int(m[2]))
        # e = tuple((r[0].strip("'"), r[1].strip("'")))
        # m = tuple((r[2].strip("'"), r[3].strip("'"), int(r[4])))
        em[e] = m
    f.close()
    return em

def dmfromem(em):
    dm = dict()
    for e in em:
        t = tuple((em[e][0], em[e][1]))
        if t not in dm:
            dm[t] = set()
        (dm[t]).add(e)
    return dm

def main():
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()

    if len(sys.argv) != 2:
        print('Usage: ', end = '')
        print(sys.argv[0], end = ' ')
        print('<input-file>')
        return

    up = readup(sys.argv[1])
    if not up:
        return

    pu = uptopu(up)

    print('Removing doms + zero-neighbour edges...')
    sys.stdout.flush()

    timeone = time.time()
    em = dict()
    dm = dict()
    seq = 0
    seq = removedominators(em, dm, up, seq)
    timetwo = time.time()

    print('done! Time taken:', timetwo - timeone)
    sys.stdout.flush()

    nedges = 0
    for u in up:
        nedges += len(up[u])

    print("Original # edges:", nedges)
    print('# dominators + zero neighbour edges removed:', seq)
    print('# remaining edges:', nedges - seq)
    print('em:')
    for e in em:
        print('\t' + str(e) + ': ' + str(em[e]))

    print('dm:')
    for d in dm:
        print('\t'+str(d)+': '+str(dm[d]))

    nzerodeg = 0
    for e in em:
        if em[e][0] < 0:
            nzerodeg += 1

    print('# edges with no neighbours:', nzerodeg)

    emfilename = sys.argv[1]+'-em.txt'
    print('Saving em into', emfilename, end=' ')
    sys.stdout.flush()
    saveem(em, emfilename)
    print('done!')
    sys.stdout.flush()

    """
    print('Reading em from', emfilename)
    sys.stdout.flush()
    rem = readem(emfilename)

    #print('em we read:')
    #for e in rem:
    #    print('\t'+str(e)+': '+str(em[e]))

    rdm = dmfromem(em)
    #print('dm from em:')
    #for d in rdm:
    #    print('\t'+str(d)+': '+str(dm[d]))

    if rem != em:
        print('BUG! rem != em')
    else:
        print('RELIEF! rem == em')

    if rdm != dm:
        print('BUG! rdm != dm')
    else:
        print('RELIEF! rdm == dm')
    """

    print('End time:', datetime.datetime.now())

if __name__ == '__main__':
    main()

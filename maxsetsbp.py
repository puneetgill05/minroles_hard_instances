#! /usr/bin/python3

import sys
import time
import os
import datetime
from pprint import pprint

import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
from colorama import Fore

prefix_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{prefix_dir}')
sys.path.append(f'{prefix_dir}/..')
# sys.path.append(f'{prefix_dir}/biclique_enumeration')
print(sys.path)

import utils
from readup import readup, uptopu, dumpup
from removedominatorsbp import removedominators, neighbours
from findcliquesbp import getedgeset, find_bicliquesbp, TTT_loop2, find_bicliquesbp2
from removedominatorsbp import saveem, readem, dmfromem, hasbeenremoved
# from utils import (getResults, make_biclique)


def maxsetsbp(em, up, pu, upfilename):
    THRESHOLD = 5000000

    print('Getting edgeset...', end='')
    sys.stdout.flush()

    edgeset = getedgeset(em, up)

    print('done!')
    sys.stdout.flush()

    cliqueset = list()
    nc = 0
    timeone = datetime.datetime.now()


    for c in find_bicliquesbp(em, up, pu, list()):
        cliqueset.append(c)
        # print('maximal biclique found: ', c)
        nc += 1
        if nc > THRESHOLD:
            print('# cliques >', THRESHOLD, ', giving up...', end='')
            sys.stdout.flush()
            break
        if not (nc % 10000):
            timetwo = datetime.datetime.now()
            print(nc, ', Time:', timetwo - timeone)
            sys.stdout.flush()

    if nc > THRESHOLD:
        print('End time:', datetime.datetime.now())
        sys.stdout.flush()
        sys.exit()

    print('# cliques:', nc)
    sys.stdout.flush()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", 42)
    env.start()

    # construct and solve ILP instance

    print('Constructing Maxsets LP...')
    sys.stdout.flush()
    constrtimeone = time.time()

    m = gp.Model("maxset", env=env)
    # m.setParam('Seed', 123)

    pc = 0
    for snum in range(len(cliqueset)):
        m.addVar(name='v_' + str(snum), vtype=GRB.BINARY)
        pc += 1
        if not (pc % 10000):
            print('Added var', pc, '/', len(cliqueset), '...')
            sys.stdout.flush()
    m.update()

    pc = 0
    obj = LinExpr()
    for u in m.getVars():
        obj.addTerms(1.0, u)
        pc += 1
        if not (pc % 1000000):
            print('Added term to objective', pc, '/', len(cliqueset), '...')
            sys.stdout.flush()
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()

    """
    #First count the num constraints for progress-printing
    print('Figuring total # constraints...')
    sys.stdout.flush()

    nconstr = 0
    nu = 0
    for u in edgeset:
        nu += 1
        if not (nu % 1000):
            print('Edge #', nu, '/', len(edgeset), '...')
            sys.stdout.flush()
        #ns = 0
        for theset in cliqueset:
            #ns += 1
            #if not (ns % 100000):
                #print('Cliqueset #', ns, '/', len(cliqueset), '...')
                #sys.stdout.flush()
            if u in theset:
                nconstr += 1
                if not (nconstr % 100000):
                    print('Constraint #:', nconstr, '...')
                    sys.stdout.flush()
    print('done! # constraints:', nconstr)
    sys.stdout.flush()
    """

    print('Adding constraints...')
    sys.stdout.flush()

    pc = 0
    nu = 0
    timeone = time.time()
    timetwo = time.time()
    for u in edgeset:
        nu += 1
        if not (nu % 1000):
            timetwo = time.time()
            print('Edge #', nu, '/', len(edgeset), '; time taken:', timetwo - timeone, '...')
            sys.stdout.flush()
            timeone = time.time()
        c = LinExpr()
        for snum, theset in enumerate(cliqueset):
            if u in theset:
                c.addTerms(1.0, m.getVarByName('v_' + str(snum)))
                pc += 1
                if not (pc % 1000000):
                    print('Added term to constraint', pc, '...')
                    sys.stdout.flush()
        m.addConstr(c >= 1, 'c_' + str(u[0]) + '_' + str(u[1]))

    m.update()

    timeone = time.time()
    print('Maxsets LP constructed, time:', timeone - constrtimeone)
    sys.stdout.flush()

    m.optimize()
    # m.write('irreducible_LP1.lp')
    # m.write('test4.lp')
    # m.write('PLAIN_small_08.lp')

    timetwo = time.time()
    print('Maxsets LP solved, time:', timetwo - timeone)
    sys.stdout.flush()

    if m.status != GRB.OPTIMAL:
        print('Weird. m.status != GRB.OPTIMAL for maxsets. Exiting...')
        sys.exit()

    print('Obj: %g' % obj.getValue())
    sys.stdout.flush()

    """
    csfile = infile+'-cs.txt'
    csf = open(csfile,'w')
    for snum, theset in enumerate(cliqueset):
        varname = 'v_'+str(snum)
        var = m.getVarByName(varname)
        if var.X:
            csf.write(str(set(theset))+'\n')
    csf.close()
    print('Solution max cliquesets written to', csfile)
    sys.stdout.flush()
    """

    """
    """
    # Update em
    print('Updating em', end='...')
    sys.stdout.flush()

    # Find highest seq num
    seq = 0
    for e in em:
        if seq < em[e][2]:
            seq = em[e][2]

    roles = list()
    for snum, theset in enumerate(cliqueset):
        varname = 'v_' + str(snum)
        var = m.getVarByName(varname)
        if not var.X:
            continue
        firstedge = None
        roles.append(theset)
        for e in theset:
            if hasbeenremoved(e, em):
                continue
            seq += 1
            if not firstedge:
                em[e] = tuple((-1, -1, seq))
                firstedge = e
            else:
                em[e] = tuple((firstedge[0], firstedge[1], seq))
    print('done!')
    sys.stdout.flush()
    """
    """

    return obj.getValue(), roles


def main():
    start_time = datetime.datetime.now()
    print('Start time:', start_time)
    sys.stdout.flush()

    if len(sys.argv) != 2:
        print('Usage: ', end='')
        print(sys.argv[0], end=' ')
        print('<input-file>')
        return

    upfilename = sys.argv[1]

    num_roles, roles = run(upfilename)
    print(f'Time taken: {(datetime.datetime.now() - start_time).total_seconds()} seconds')

    print('UP:')
    up = readup(upfilename)

    pprint(up)
    utils.getResults(roles)


def run(upfilename: str, remove_dominators: bool=True):
    # up = mapup(up, sys.argv[1])
    up = readup(upfilename)
    if not up:
        return

    pu = uptopu(up)

    nedges = 0
    for u in up:
        nedges += len(up[u])

    print('Total # edges:', nedges)
    sys.stdout.flush()

    timeone = time.time()

    seq = 0
    em = dict()
    dm = dict()

    """
    """
    # upfilename = upfilename.replace('.txt', '')
    # emfilename = upfilename.replace('.txt', '') + '-em.txt'
    emfilename = upfilename + '-em.txt'
    if remove_dominators:
        print('Removing dominators:', remove_dominators)
        if not os.path.isfile(emfilename):
            print('Removing doms + zero-neighbour edges...')
            sys.stdout.flush()
            seq = removedominators(em, dm, up, seq)
            timetwo = time.time()
            print('done! Time taken:', timetwo - timeone)
            sys.stdout.flush()
            print('Saving em to', emfilename, end=' ')
            sys.stdout.flush()
            saveem(em, emfilename)
            print('done!')
            sys.stdout.flush()
        else:
            print('Reading em from', emfilename, end=' ')
            sys.stdout.flush()
            em = readem(emfilename)
            print('done!')
            sys.stdout.flush()
            print('Determining dm and seq', end=' ')
            sys.stdout.flush()
            dm = dmfromem(em)
            for e in em:
                if seq < em[e][2]:
                    seq = em[e][2]
            print('done!')
            sys.stdout.flush()
    else:
        print('NOT RUNNING REMOVE DOMINATORS')
    print("Original # edges:", nedges)
    # em = dict()

    # Need to count doms and zero-neigh in my up set
    nmydom = 0
    nmyzerodeg = 0
    for e in em:
        u = e[0]
        p = e[1]

        if u in up and p in up[u]:
            if (isinstance(em[e][0], int) or em[e][0].isdigit()) and em[e][0] < 0:
                nmyzerodeg += 1
            else:
                nmydom += 1

    print('# my dominators + zero neighbour edges removed:', nmydom + nmyzerodeg)
    print('# edges with -1 annotation:', nmyzerodeg)
    """
    """

    obj = 0
    roles = list()
    # if nmydom + nmyzerodeg < nedges:
    obj, roles = maxsetsbp(em, up, pu, upfilename)
    # save em
    print('Saving em to', emfilename, end='...')
    sys.stdout.flush()
    saveem(em, emfilename)
    print('done!')
    roles = form_roles(em, up, [])
    # else:
    #     print('Nothing more to be done.')
    #     sys.stdout.flush()
    #     num_roles = 0
    #     for e in em:
    #         val = em[e]
    #         if val[0] == -1 and val[1] == -1:
    #             num_roles += 1
    #     roles = form_roles(em, up, [])

    roles_dict = dict()
    for r in range(len(roles)):
        if r not in roles_dict:
            roles_dict[r] = set()
        for e in roles[r]:
            roles_dict[r].add(e[0])
            roles_dict[r].add(e[1])

    num_edges = sum(len(roles_dict[r]) for r in roles_dict)
    # print('Final solution:', nmyzerodeg + obj)
    print(Fore.GREEN + 'Number of roles: ' + str(len(roles)))
    print(Fore.GREEN + 'Number of edges: ' + str(num_edges))
    # if utils.check_roles(roles, up):
    #     print(Fore.GREEN + 'Final roles check satisfied')
    # else:
    #     print(Fore.RED + 'Final roles check satisfied')
    print(Fore.RESET)

    print('End time:', datetime.datetime.now())
    sys.stdout.flush()
    return nmyzerodeg + obj, roles


def assign_groups(em: dict[tuple[int, int], tuple[int, int], tuple[int, int]]):
    SENTINEL = (-1, -1)
    # node -> terminal representative
    terminal_of: dict[tuple[int, int], tuple[int, int]] = {}

    # terminal representative -> list of nodes in that group
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for start in em:
        if start in terminal_of:
            continue

        path = []
        cur = start
        seen_in_this_walk = set()

        while True:
            if cur in terminal_of:
                # already resolved earlier
                terminal = terminal_of[cur]
                break

            if cur in seen_in_this_walk:
                raise ValueError(f"Cycle detected involving {cur}")

            if cur not in em:
                # raise KeyError(f"Edge {cur} not found in em")
                print(f"Edge {cur} not found in em")

            seen_in_this_walk.add(cur)
            path.append(cur)
            # if cur in em:
            nxt = em[cur][:-1]

            if nxt == SENTINEL:
                # current node is terminal
                terminal = cur
                break
            cur = nxt
            # else:
            #     terminal = cur
            #     break
        # assign every node on the path to the same terminal
        for node in path:
            terminal_of[node] = terminal

        groups.setdefault(terminal, []).extend(path)
    return terminal_of, groups


def form_roles(em, up, existing_roles):
    # print('UP:')
    # pprint(up)
    _, roles = assign_groups(em)
    # pprint(roles)

    new_roles = []
    for r in roles:
        new_roles.append(set(roles[r]))



    # zero_degree_edges = set()
    # dominator_edges = set()
    # all_edges = set()
    # for u in up:
    #     for p in up[u]:
    #         all_edges.add((u, p))
    # for e in em:
    #     u = e[0]
    #     p = e[1]
    #     if u in up and p in up[u]:
    #         if (isinstance(em[e][0], int) or em[e][0].isdigit()) and em[e][0] < 0:
    #             zero_degree_edges.add(e)
    #         else:
    #             dominator_edges.add(e)
    #
    # roles = list()
    # for r in existing_roles:
    #     roles.append(set(r))
    # # create a role for zero degree edges
    # for e in zero_degree_edges:
    #     new_role = {e}
    #     subset_found = False
    #     for role in roles:
    #         if set(new_role).issubset(set(role)):
    #             subset_found = True
    #     if not subset_found:
    #         roles.append(new_role)
    # # if edge d dominates edge e, then add the dominator edge d to the role which has edge e
    # edges_covered_by_roles = set()
    #
    # max_edges = 1000000
    # e_ctr = 0
    # while True and e_ctr < max_edges:
    #     for d in dominator_edges:
    #         e = (em[d][0], em[d][1])
    #         for role in roles:
    #             if isinstance(role, list):
    #                 role = set(role)
    #             if e in role:
    #                 role.add(d)
    #     new_roles = list()
    #     for role in roles:
    #         made_role = set(utils.make_biclique(set(role)))
    #         subset_found = False
    #         for new_role in new_roles:
    #             if set(made_role).issubset(set(new_role)):
    #                 subset_found = True
    #         if not subset_found:
    #             new_roles.append(made_role)
    #
    #     for role in new_roles:
    #         for e in role:
    #             edges_covered_by_roles.add(e)
    #     # ensure no edge is missing
    #     print('Set diff: ', len(all_edges.difference(edges_covered_by_roles)))
    #     if len(all_edges.difference(edges_covered_by_roles)) == 0:
    #         break
    #     e_ctr += 1
    #     if e_ctr % 10000 == 0:
    #         print(e_ctr)
    return new_roles


if __name__ == '__main__':
    # inputfilename = '/home/puneet/Projects/minedgerolemining/minedgerolemining/inputsup/test1.txt'
    # em = readem(f'{inputfilename}-em.txt')
    # up = readup(inputfilename)
    main()
    # roles = form_roles(em, up, [])
    # pprint(len(roles))

import sys
sys.path.append('/home/james/Dropbox (MIT)/All_Resourceful/dag_planner/dag-plan')
sys.path.append('/home/james/Dropbox (MIT)/All_Resourceful/localgateway')
#sys.path.append('../../../dag_planner/dag-plan')
#sys.path.append('/home/jjlong/localgateway')
from app.dag_solver import solve_DAG
from app.batt_dag_solver import solve_batt_DAG
from app.views import inted
import numpy as np
import random
import math
import functools
from scipy.stats import rv_discrete
from beeview_gateway import parse_dag_stats as pds
import copy
from multiprocessing import Pool

def tuplify(d):
    return tuple([(k,v) for k,v in d.items()])
import json
def parser(d):
    reformatted= pds.reformat(d['sol'],d['graph'])
    nodes_translated = pds.translate_nodeweights(d['px'],
                                                 reformatted)
    parsed = pds.translate_edgeweights(d['bw'],nodes_translated)
    return parsed
def summer(parsed_d):
    return {g:pds.get_node_total(i) for g,i in parsed_d}
def to_battery(active_mA,t):
    #takes the active milliAmp consumption
    #rate and for and an active time
    #in milliseconds t, converts to a consumed mAh
    seconds = t/1000
    mAseconds = active_mA*seconds #milliAmp seconds
    return mAseconds/(60*60)
def sum_to_consumed(draw,summed):
    return {k: to_battery(draw,v) 
            for k,v in summed.items()
            if k!=0}
def dict_to_tuple(d):
    tuplified_inner = {k:tuplify(v) for k, v in d.items()}
    return tuplify(tuplified_inner) 
def tuple_to_dict(t):
    return {k:dict(v) for k, v in dict(t).items()}
def gen_times(interval, numdays):
    every_day = 24/interval
    numintervals = int(numdays*every_day)
    return [i*interval for i in range(1,numintervals+1)]
def poisson_pmf(mean_per_unittime, t, numevents):
    r = mean_per_unittime
    return ((r*t)**numevents)*math.exp(-r*t)/math.factorial(numevents)

def get_batt_drawdown(tpx,tbw,code, tbatt):
    px = dict(tpx)
    bw = tuple_to_dict(tbw)
    batteries = dict(tbatt)
    active_draw = 56+33+50#pyboard+digimesh+accel
    sol = solve_batt_DAG(code, None, px,batteries, ack=580, bw=bw)
    d = json.loads(json.dumps(sol))
    time_taken = summer(parser(d))
    return sum_to_consumed(active_draw, time_taken)
@functools.lru_cache(maxsize=512)
def get_drawdown(tpx,tbw,code, batteries):
    px = dict(tpx)
    bw = tuple_to_dict(tbw)
    active_draw = 56+33+50#pyboard+digimesh+accel
    sol = solve_DAG(code, None, px, ack=580, bw=bw)
    d = json.loads(json.dumps(sol))
    time_taken = summer(parser(d))
    return sum_to_consumed(active_draw, time_taken)
def down_calcer(previous, drawdowns):
    downcalc = lambda previous, num_down: max(0, previous-num_down)
    return {k:downcalc(previous[k], v) for k,v in drawdowns.items()}
def poisson_sample(intensity, t, numsims=1):
    xs = range(100)
    pmf = [poisson_pmf(intensity, t, x) for x in xs]
    sample=rv_discrete(values=(xs,pmf)).rvs(size=numsims)
    return sample
def poisson_step(intensity, previous, time_elapsed, units=1):
    num_down = poisson_sample(intensity, time_elapsed, numsims=1)[0]
    return max(units, previous-num_down*units)

def poisson_evolution(intensities, px, elapsed):
    new_pxs =  {k: poisson_step(intensities[k], v, elapsed, units=0.005)
           for k,v in px.items()}
    new_pxs[0]=1
    return new_pxs
def active_step(drawdown_func, code, bw,intensities, previousval, previouspx, elapsed):
    new_px = poisson_evolution(intensities, previouspx, elapsed)
    tpx = tuplify(new_px)
    tbw = dict_to_tuple(bw)
    tval = tuplify(previousval)
    drawdowns = drawdown_func(tpx, tbw, code, tval)
    return down_calcer(previousval, drawdowns), new_px
def passive_step(previous, time_elapsed):
    passive = 0.3
    drawdown = {k: to_battery(passive,1000*time_elapsed*3600) 
                for k in previous}
    return down_calcer(previous, drawdown)
def markov_walker(stepper, accumed, this_t):
    """accumed is a list of (t, value) tuples"""
    print('stepping at time: ', this_t)
    previous_t, previous_val_shallow, previous_px_shallow = accumed[-1]
    previous_val = copy.deepcopy(previous_val_shallow)
    previous_px = copy.deepcopy(previous_px_shallow)
    time_elapsed = this_t - previous_t
    passive_val = passive_step(previous_val, time_elapsed)
    accumed.append((this_t, passive_val, previous_px))
    this_val, this_px = stepper(passive_val, previous_px, time_elapsed)
    accumed.append((this_t, this_val, this_px))
    return accumed

def get_walker(stepper):
    return functools.partial(markov_walker, stepper)

def initialise_battery(node, val):
    return val if node not in [0,'0'] else np.inf

if __name__ == '__main__':
    bw = {'0': {'0': 0.0,
  '15': 53.25,
  '17': 60.19,
  '18': 63.31,
  '22': 50.7,
  '31': 50.11,
  '32': 51.12,
  '37': 63.31,
  '39': 50.37,
  '40': 51.66,
  '41': 53.25,
  '43': 52.36,
  '46': 99.33,
  '49': 49.92,
  '53': 98.43,
  '55': 60.19,
  '56': 51.12,
  '58': 57.77,
  '61': 53.25,
  '63': 99.58,
  '64': 98.43,
  '68': 50.37,
  '69': 99.91,
  '95': 99.33,
  '96': 55.87},
 '15': {'0': 349.69,
  '15': 0.0,
  '17': 246.07,
  '18': 147.65,
  '22': 147.64,
  '31': 345.05,
  '32': 49.22,
  '37': 246.07,
  '39': 246.07,
  '40': 196.86,
  '41': 196.86,
  '43': 246.07,
  '46': 147.66,
  '49': 295.29,
  '53': 246.07,
  '55': 295.29,
  '56': 295.29,
  '58': 295.29,
  '61': 147.64,
  '63': 295.29,
  '64': 147.65,
  '68': 246.08,
  '69': 196.86,
  '95': 344.51,
  '96': 98.43},
 '17': {'0': 251.26,
  '15': 98.43,
  '17': 0.0,
  '18': 49.22,
  '22': 147.64,
  '31': 246.62,
  '32': 98.44,
  '37': 147.64,
  '39': 147.64,
  '40': 196.86,
  '41': 98.43,
  '43': 49.21,
  '46': 147.66,
  '49': 49.21,
  '53': 147.64,
  '55': 98.43,
  '56': 196.86,
  '58': 98.43,
  '61': 98.43,
  '63': 196.86,
  '64': 49.21,
  '68': 147.65,
  '69': 98.43,
  '95': 147.67,
  '96': 98.43},
 '18': {'0': 300.48,
  '15': 49.21,
  '17': 196.86,
  '18': 0.0,
  '22': 147.65,
  '31': 295.84,
  '32': 49.22,
  '37': 196.86,
  '39': 196.86,
  '40': 196.86,
  '41': 49.21,
  '43': 246.07,
  '46': 147.67,
  '49': 246.07,
  '53': 196.86,
  '55': 295.29,
  '56': 246.08,
  '58': 246.07,
  '61': 49.22,
  '63': 246.08,
  '64': 98.43,
  '68': 196.86,
  '69': 147.65,
  '95': 295.29,
  '96': 98.44},
 '22': {'0': 202.05,
  '15': 196.85,
  '17': 98.43,
  '18': 147.64,
  '22': 0.0,
  '31': 197.41,
  '32': 196.86,
  '37': 98.43,
  '39': 98.43,
  '40': 49.21,
  '41': 147.64,
  '43': 98.43,
  '46': 147.66,
  '49': 147.64,
  '53': 98.43,
  '55': 147.65,
  '56': 147.65,
  '58': 147.64,
  '61': 98.43,
  '63': 147.65,
  '64': 147.64,
  '68': 98.43,
  '69': 49.21,
  '95': 196.86,
  '96': 98.43},
 '31': {'0': 152.83,
  '15': 98.43,
  '17': 49.21,
  '18': 98.43,
  '22': 98.43,
  '31': 0.0,
  '32': 98.43,
  '37': 98.43,
  '39': 49.21,
  '40': 49.21,
  '41': 49.21,
  '43': 49.21,
  '46': 49.21,
  '49': 98.43,
  '53': 49.21,
  '55': 98.43,
  '56': 49.21,
  '58': 49.21,
  '61': 49.21,
  '63': 98.43,
  '64': 98.43,
  '68': 49.21,
  '69': 98.43,
  '95': 49.21,
  '96': 98.43},
 '32': {'0': 300.48,
  '15': 147.64,
  '17': 196.85,
  '18': 98.43,
  '22': 98.43,
  '31': 295.83,
  '32': 0.0,
  '37': 196.85,
  '39': 196.86,
  '40': 147.64,
  '41': 147.64,
  '43': 196.86,
  '46': 98.44,
  '49': 246.07,
  '53': 196.85,
  '55': 246.07,
  '56': 246.07,
  '58': 246.07,
  '61': 98.43,
  '63': 246.07,
  '64': 98.43,
  '68': 196.86,
  '69': 147.64,
  '95': 295.29,
  '96': 49.21},
 '37': {'0': 300.48,
  '15': 147.65,
  '17': 49.22,
  '18': 98.43,
  '22': 196.86,
  '31': 295.84,
  '32': 147.66,
  '37': 0.0,
  '39': 196.86,
  '40': 246.07,
  '41': 147.65,
  '43': 98.43,
  '46': 196.87,
  '49': 98.43,
  '53': 196.86,
  '55': 147.65,
  '56': 246.08,
  '58': 147.64,
  '61': 147.65,
  '63': 246.08,
  '64': 98.43,
  '68': 196.86,
  '69': 147.64,
  '95': 196.88,
  '96': 147.64},
 '39': {'0': 152.83,
  '15': 98.43,
  '17': 98.43,
  '18': 49.21,
  '22': 147.64,
  '31': 148.19,
  '32': 98.43,
  '37': 49.21,
  '39': 0.0,
  '40': 147.64,
  '41': 49.21,
  '43': 147.64,
  '46': 147.64,
  '49': 49.21,
  '53': 147.64,
  '55': 196.86,
  '56': 98.43,
  '58': 49.21,
  '61': 98.43,
  '63': 98.43,
  '64': 98.43,
  '68': 49.21,
  '69': 147.65,
  '95': 147.64,
  '96': 98.43},
 '40': {'0': 202.04,
  '15': 147.64,
  '17': 147.64,
  '18': 98.43,
  '22': 98.43,
  '31': 197.4,
  '32': 147.65,
  '37': 98.43,
  '39': 196.86,
  '40': 0.0,
  '41': 147.64,
  '43': 49.22,
  '46': 98.44,
  '49': 196.86,
  '53': 49.21,
  '55': 98.43,
  '56': 147.64,
  '58': 98.43,
  '61': 98.43,
  '63': 147.64,
  '64': 98.43,
  '68': 98.43,
  '69': 147.64,
  '95': 147.67,
  '96': 49.21},
 '41': {'0': 251.27,
  '15': 147.65,
  '17': 147.65,
  '18': 98.44,
  '22': 147.64,
  '31': 246.63,
  '32': 49.21,
  '37': 147.65,
  '39': 147.65,
  '40': 196.86,
  '41': 0.0,
  '43': 196.86,
  '46': 147.66,
  '49': 196.86,
  '53': 147.65,
  '55': 246.08,
  '56': 196.86,
  '58': 196.86,
  '61': 98.44,
  '63': 196.87,
  '64': 49.22,
  '68': 147.65,
  '69': 98.43,
  '95': 246.08,
  '96': 98.43},
 '43': {'0': 202.05,
  '15': 147.64,
  '17': 147.67,
  '18': 98.43,
  '22': 98.43,
  '31': 197.41,
  '32': 147.65,
  '37': 147.69,
  '39': 147.67,
  '40': 147.64,
  '41': 147.64,
  '43': 0.0,
  '46': 98.44,
  '49': 147.67,
  '53': 147.67,
  '55': 49.22,
  '56': 147.65,
  '58': 49.21,
  '61': 98.43,
  '63': 147.65,
  '64': 98.43,
  '68': 98.44,
  '69': 147.64,
  '95': 98.45,
  '96': 49.21},
 '46': {'0': 202.05,
  '15': 196.86,
  '17': 98.43,
  '18': 147.65,
  '22': 147.65,
  '31': 197.41,
  '32': 196.86,
  '37': 98.43,
  '39': 98.43,
  '40': 196.86,
  '41': 147.65,
  '43': 147.65,
  '46': 0.0,
  '49': 147.65,
  '53': 98.43,
  '55': 196.86,
  '56': 147.65,
  '58': 147.65,
  '61': 98.43,
  '63': 147.65,
  '64': 147.65,
  '68': 98.44,
  '69': 49.22,
  '95': 196.87,
  '96': 98.43},
 '49': {'0': 300.48,
  '15': 98.43,
  '17': 196.86,
  '18': 49.22,
  '22': 98.43,
  '31': 295.83,
  '32': 98.44,
  '37': 196.86,
  '39': 196.86,
  '40': 147.64,
  '41': 98.43,
  '43': 196.86,
  '46': 98.44,
  '49': 0.0,
  '53': 196.86,
  '55': 246.07,
  '56': 246.07,
  '58': 246.07,
  '61': 98.43,
  '63': 246.07,
  '64': 98.43,
  '68': 196.86,
  '69': 147.64,
  '95': 295.29,
  '96': 49.21},
 '53': {'0': 152.83,
  '15': 147.64,
  '17': 98.43,
  '18': 98.43,
  '22': 49.22,
  '31': 148.19,
  '32': 147.65,
  '37': 49.22,
  '39': 147.64,
  '40': 98.43,
  '41': 147.64,
  '43': 147.64,
  '46': 98.44,
  '49': 147.64,
  '53': 0.0,
  '55': 196.86,
  '56': 98.43,
  '58': 49.22,
  '61': 98.43,
  '63': 98.43,
  '64': 98.43,
  '68': 49.21,
  '69': 98.43,
  '95': 147.64,
  '96': 49.21},
 '55': {'0': 152.83,
  '15': 98.45,
  '17': 98.45,
  '18': 98.45,
  '22': 98.45,
  '31': 148.19,
  '32': 147.66,
  '37': 98.47,
  '39': 98.45,
  '40': 98.46,
  '41': 147.66,
  '43': 98.45,
  '46': 98.45,
  '49': 98.45,
  '53': 98.45,
  '55': 0.0,
  '56': 98.43,
  '58': 98.45,
  '61': 98.43,
  '63': 98.43,
  '64': 147.65,
  '68': 49.22,
  '69': 98.45,
  '95': 49.23,
  '96': 98.45},
 '56': {'0': 251.27,
  '15': 147.64,
  '17': 147.65,
  '18': 98.43,
  '22': 98.43,
  '31': 246.63,
  '32': 147.65,
  '37': 147.65,
  '39': 147.65,
  '40': 147.64,
  '41': 147.64,
  '43': 196.86,
  '46': 98.44,
  '49': 196.86,
  '53': 147.65,
  '55': 246.08,
  '56': 0.0,
  '58': 196.86,
  '61': 98.43,
  '63': 196.87,
  '64': 49.22,
  '68': 147.65,
  '69': 98.43,
  '95': 246.08,
  '96': 49.21},
 '58': {'0': 300.48,
  '15': 147.64,
  '17': 196.86,
  '18': 98.43,
  '22': 98.43,
  '31': 295.84,
  '32': 147.65,
  '37': 196.86,
  '39': 196.86,
  '40': 147.64,
  '41': 147.64,
  '43': 196.86,
  '46': 98.44,
  '49': 246.07,
  '53': 196.86,
  '55': 246.08,
  '56': 246.08,
  '58': 0.0,
  '61': 98.43,
  '63': 246.08,
  '64': 98.43,
  '68': 196.86,
  '69': 147.64,
  '95': 295.29,
  '96': 49.21},
 '61': {'61': 0.0},
 '63': {'0': 54.4,
  '15': 49.21,
  '17': 98.43,
  '18': 98.43,
  '22': 49.29,
  '31': 49.76,
  '32': 49.31,
  '37': 49.22,
  '39': 49.21,
  '40': 49.22,
  '41': 49.23,
  '43': 98.43,
  '46': 49.22,
  '49': 98.43,
  '53': 49.22,
  '55': 114.59,
  '56': 49.21,
  '58': 98.43,
  '61': 98.43,
  '63': 0.0,
  '64': 49.23,
  '68': 98.43,
  '69': 98.43,
  '95': 49.22,
  '96': 49.22},
 '64': {'0': 202.05,
  '15': 98.43,
  '17': 98.43,
  '18': 49.22,
  '22': 147.64,
  '31': 197.41,
  '32': 98.44,
  '37': 98.43,
  '39': 98.43,
  '40': 196.86,
  '41': 98.43,
  '43': 147.64,
  '46': 147.66,
  '49': 147.64,
  '53': 98.43,
  '55': 196.86,
  '56': 147.65,
  '58': 147.64,
  '61': 49.22,
  '63': 147.65,
  '64': 0.0,
  '68': 98.43,
  '69': 49.21,
  '95': 196.86,
  '96': 98.43},
 '68': {'0': 103.61,
  '15': 98.43,
  '17': 147.64,
  '18': 147.64,
  '22': 98.5,
  '31': 98.97,
  '32': 98.52,
  '37': 98.43,
  '39': 98.43,
  '40': 98.43,
  '41': 98.45,
  '43': 147.64,
  '46': 98.43,
  '49': 147.64,
  '53': 98.43,
  '55': 163.81,
  '56': 49.21,
  '58': 147.64,
  '61': 49.21,
  '63': 49.21,
  '64': 98.43,
  '68': 0.0,
  '69': 147.64,
  '95': 98.43,
  '96': 98.43},
 '69': {'0': 152.83,
  '15': 147.64,
  '17': 49.21,
  '18': 98.43,
  '22': 98.43,
  '31': 148.19,
  '32': 147.64,
  '37': 49.21,
  '39': 49.21,
  '40': 147.64,
  '41': 98.43,
  '43': 98.43,
  '46': 98.44,
  '49': 98.43,
  '53': 49.21,
  '55': 147.65,
  '56': 98.43,
  '58': 98.43,
  '61': 49.21,
  '63': 98.43,
  '64': 98.43,
  '68': 49.22,
  '69': 0.0,
  '95': 147.65,
  '96': 49.21},
 '95': {'0': 152.83,
  '15': 49.21,
  '17': 49.21,
  '18': 49.21,
  '22': 49.22,
  '31': 148.19,
  '32': 98.43,
  '37': 49.23,
  '39': 49.21,
  '40': 49.22,
  '41': 98.43,
  '43': 49.21,
  '46': 49.21,
  '49': 49.21,
  '53': 49.22,
  '55': 98.43,
  '56': 49.92,
  '58': 49.22,
  '61': 49.23,
  '63': 98.43,
  '64': 98.43,
  '68': 49.22,
  '69': 49.22,
  '95': 0.0,
  '96': 49.21},
 '96': {'0': 251.26,
  '15': 98.43,
  '17': 147.64,
  '18': 49.22,
  '22': 49.21,
  '31': 246.62,
  '32': 98.44,
  '37': 147.64,
  '39': 147.64,
  '40': 98.43,
  '41': 98.43,
  '43': 147.64,
  '46': 49.23,
  '49': 196.85,
  '53': 147.64,
  '55': 196.86,
  '56': 196.86,
  '58': 196.86,
  '61': 49.21,
  '63': 196.86,
  '64': 49.22,
  '68': 147.65,
  '69': 98.43,
  '95': 246.08,
  '96': 0.0}}
    fmatted_bw = inted({k:inted(v) for k,v in bw.items()})
    pxval = lambda k: 0.05 if k!=0 else 1
    px = {k:pxval(k) for k in fmatted_bw}
    intel_code = """class SenseReduce:
    def __init__(self):
        self.sensenodes = [[63],[41],[15],[95],[96],[53],[55],[31]]
        self.mapnodes = [[63],[41],[15],[95],[96],[53],[55],[31]]
        self.l=512
        self.reducenodes = [[63,41,15,95,96,53,55,31]]
    def sampler(self,node):
        acc = yield from node.testaccel(512)
        return (node.ID,acc)
    @slowdown()
    def mapper(self,node,d):
        nodeid, data = d
        for ax in data:
            ftpeak = np.fft(data[ax])[6]
            c = lambda d: (round(d.real,6),round(d.imag,6))
            nodeax = str(nodeid)+ax
            yield(0,(nodeax,c(ftpeak)))
    @slowdown()
    def reducer(self,node,k,vs):
        ws = [complex(*i[1]) for i in vs]
        G = np.spectral_mat(ws)
        eig = np.pagerank(G)
        c = lambda d: (round(d.real,2),round(d.imag,2))
        ms = [(vs[idx][0],c(el)) for idx,el in enumerate(eig)]
        yield(k,ms)"""
    naive_code = """class SenseReduce:
    def __init__(self):
        self.sensenodes = [[63],[41],[15],[95],[96],[53],[55],[31]]
        self.mapnodes = [[0],[0],[0],[0],[0],[0],[0],[0]]
        self.l=512
        self.reducenodes = [[0]]
    def sampler(self,node):
        acc = yield from node.testaccel(512)
        return (node.ID,acc)
    @slowdown()
    def mapper(self,node,d):
        nodeid, data = d
        for ax in data:
            ftpeak = np.fft(data[ax])[6]
            c = lambda d: (round(d.real,6),round(d.imag,6))
            nodeax = str(nodeid)+ax
            yield(0,(nodeax,c(ftpeak)))
    @slowdown()
    def reducer(self,node,k,vs):
        ws = [complex(*i[1]) for i in vs]
        G = np.spectral_mat(ws)
        eig = np.pagerank(G)
        c = lambda d: (round(d.real,2),round(d.imag,2))
        ms = [(vs[idx][0],c(el)) for idx,el in enumerate(eig)]
        yield(k,ms)"""
    interval = 6#hours
    days = 180
    times = gen_times(interval,days)
    get_intensities = lambda px: {node:random.gauss(0.01,0.001) for node in px}
    initial_conditions = lambda val,px : [(0, {str(k):initialise_battery(k,val) for k in px}, px)]
    our_stepper = functools.partial(active_step, get_drawdown,
                                    intel_code,fmatted_bw,
                                    get_intensities(px))
    naive_stepper = functools.partial(active_step, get_drawdown,
                                    naive_code, fmatted_bw,
                                    get_intensities(px))
    batt_stepper = functools.partial(active_step, get_batt_drawdown,
                                    intel_code, fmatted_bw,
                                    get_intensities(px))
    def run_and_write(stepper, times,px, fname):
        res = functools.reduce(get_walker(stepper), times, initial_conditions(600,px))
        with open(fname,'a') as f:
            f.write(json.dumps(res))
        return {fname:res}
    import os
    os.nice(-20)
    run_and_write(batt_stepper, times, px, 'batt_result_8_lowcapacity.json')
    #with Pool(processes=1) as pool:  # start 4 worker processes
    #    #arg_list = [(our_stepper,times, px, 'intel_result_8_lowcapacity.json'),
    #    #            (naive_stepper, times, px, 'naive_result_8_lowcapacity.json'),
    #    #            (batt_stepper, times, px, 'batt_result_8_lowcapacity.json')]
    #    arg_list = [(batt_stepper, times, px, 'batt_result_8_lowcapacity.json')]
    #    result = pool.starmap(run_and_write, arg_list)

''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
import utils
from utils import load_datasets, load_nav_graphs, ndtw_graphload, DTW
from agent import BaseAgent
from ipdb import set_trace

class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        with open('data_v2/objpos.json','r') as f:
                self.objpos = json.load(f)

        for split in splits:
            for item in load_datasets([split]):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs, self.vppos = load_nav_graphs(self.scans, rtnpos=True)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        self.objProposals, self.obj2viewpoint = utils.loadObjProposals(self.objpos,self.vppos)

        # self.ndtw_criterion = {}
        # scan_gts_dir = '/home/yicong/research/selfmonitoring-agent/tasks/Env-back/data_v2id_paths.json'
        # with open(scan_gts_dir) as f_:
        #     self.scan_gts = json.load(f_)
        # all_scan_ids = []
        # for key in self.scan_gts:
        #     path_scan_id = self.scan_gts[key][0]
        #     if path_scan_id not in all_scan_ids:
        #         all_scan_ids.append(path_scan_id)
        #         ndtw_graph = ndtw_graphload(path_scan_id)
        #         self.ndtw_criterion[path_scan_id] = DTW(ndtw_graph)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path, ref_objId):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        gt = self.gt[instr_id[:-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        # self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        # self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)

        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )

        # REF sucess or not
        if ref_objId == str(gt['objId']):
            self.scores['rgs'].append(1)
        else:
            self.scores['rgs'].append(0)
        # navigation - success or not
        end_viewpoint_id = gt['scan'] + '_' + final_position
        if self.objProposals.__contains__(end_viewpoint_id):
            if str(gt['objId']) in self.objProposals[end_viewpoint_id]['objId']:
                self.scores['visible'].append(1)
            else:
                self.scores['visible'].append(0)
        else:
            self.scores['visible'].append(0)
        # navigation - oracle success or not
        oracle_succ = 0
        for passvp in path:
            oracle_viewpoint_id = gt['scan'] + '_' + passvp[0]
            if self.objProposals.__contains__(oracle_viewpoint_id):
                if str(gt['objId']) in self.objProposals[oracle_viewpoint_id]['objId']:
                    oracle_succ = 1
                    break
        self.scores['oracle_visible'].append(oracle_succ)


        # # if self.scores['nav_errors'][-1] < self.error_margin:
        # # print('item', item)
        # ndtw_path = [k[0] for k in item['trajectory']]
        # # print('path', ndtw_path)
        #
        # path_id = item['instr_id'][:-2]
        # # print('path id', path_id)
        # path_scan_id, path_ref = self.scan_gts[path_id]
        # # print('path_scan_id', path_scan_id)
        # # print('path_ref', path_ref)
        #
        # path_act = []
        # for jdx, pid in enumerate(ndtw_path):
        #     if jdx != 0:
        #         if pid != path_act[-1]:
        #             path_act.append(pid)
        #     else:
        #         path_act.append(pid)
        # # print('path act', path_act)
        #
        # ndtw_score = self.ndtw_criterion[path_scan_id](path_act, path_ref, metric='ndtw')
        # ndtw_scores.append(ndtw_score)
        # print('nDTW score: ', np.average(ndtw_scores))

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file

        print('result length', len(results))
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'], item['ref'])

        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                           % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['visible']) == len(self.instr_ids)

        score_summary = {
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        end_successes = sum(self.scores['visible'])
        score_summary['success_rate'] = float(end_successes) / float(len(self.scores['visible']))
        oracle_successes = sum(self.scores['oracle_visible'])
        score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_visible']))

        spl = [float(visible == 1) * l / max(l, p, 0.01)
            for visible, p, l in
            zip(self.scores['visible'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)

        #set_trace()

#        assert len(self.scores['rgs']) == len(self.instr_ids)
        num_rgs = sum(self.scores['rgs'])
        score_summary['rgs'] = float(num_rgs)/float(len(self.scores['rgs']))

        rgspl = [float(rgsi == 1) * l / max(l, p)
            for rgsi, p, l in
            zip(self.scores['rgs'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['rgspl'] = np.average(rgspl)

        return score_summary, self.scores

    def bleu_score(self, path2inst):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            candidates.append([self.tok.index_to_word[word_id] for word_id in inst])

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions


RESULT_DIR = 'tasks/R2R/results/'

def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    eval_simple_agents()

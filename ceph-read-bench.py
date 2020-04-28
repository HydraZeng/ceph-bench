#!/usr/bin/python3

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import time
from random import shuffle
from scipy.stats import zipf
import rados

def init(conf, keyring):
    cluster = rados.Rados(conffile=conf, conf = dict(keyring=keyring))
    cluster.connect()
    return cluster

def prepare_objs(ioctx, reads_num, use_zipf, zipf_parm):
    ioctx.require_ioctx_open()
    cluster_objects = list(ioctx.list_objects())

    objs = []
    count = 0
    if use_zipf:
        objs_p = [zipf.pmf(i, zipf_parm) for i in range(1, len(cluster_objects)+1)]
        objs_p /= sum(objs_p)

        objs_c = []
        for p in objs_p:
            c = int(p*reads_num) + 1 if count < reads_num else 0
            objs_c.append(c)
            count += c

        shuffle(objs_c)
        
        for i, obj in enumerate(cluster_objects):
            key = 0
            length = 0
            for j in range(objs_c[i]):
                if j == 0:
                    key = obj.key
                    length = obj.stat()[0]
                objs.append(dict(key=key,len=length))

    else:
        for obj in cluster_objects:
            objs.append(dict(key=obj.key,len=obj.stat()[0]))
            count += 1
            if count == reads_num:
                return objs
        obj_num = count
        while count < reads_num:
            idx = count % obj_num
            objs.append(dict(key=objs[idx]['key'],len=objs[idx]['len']))
            count += 1
    
    shuffle(objs)
    return objs

def obj_aio_read(tid, ioctx, read_res, obj):

    def aio_read_complete(_, data):
        read_res[tid]['end'] = time.time()
        read_res[tid]['len'] = len(data)
        read_res[tid]['finish'] = True

    read_res[tid]['start'] = time.time()
    completion = ioctx.aio_read(obj['key'], obj['len'], 0, aio_read_complete)
    return completion

def wait_finish(handlers):
    for i, handler in enumerate(handlers):
        comp = handler.result()
        comp.wait_for_complete_and_cb()

def print_statistics(bw_mbs, iops, lat):
    print('========== BW ==========')
    print('min bw: {:.2f} MB/s'.format(min(bw_mbs)))
    print('max bw: {:.2f} MB/s'.format(max(bw_mbs)))
    print('avg bw: {:.2f} MB/s'.format(sum(bw_mbs)/len(bw_mbs)))

    print('========= IOPS =========')
    print('min iops: {}'.format(min(iops)))
    print('max iops: {}'.format(max(iops)))
    print('avg iops: {:.2f}'.format(sum(iops)/len(iops)))

    print('======== LATENCY =======')
    print('min lat: {:.6f} s'.format(min(lat)))
    print('max lat: {:.6f} s'.format(max(lat)))
    print('avg lat: {:.6f} s'.format(sum(lat)/len(lat)))

def collect_and_print_statistics(res):
    bw_mbs = []
    lat = []
    base_t = 0
    end = []

    for r in res:
        if not r['finish']:
            continue
        base_t = r['start'] if base_t == 0 or r['start'] < base_t else base_t
        end.append(r['end'])
        bw_mbs.append(r['len']/(r['end']-r['start'])/(1000*1000))
        lat.append(r['end']-r['start'])

    if len(bw_mbs) == 0:
        bw_mbs.append(0)
        lat.append(0)
        iops = [0]
    else:
        total_lat = int(max(end) - base_t) + 1
        iops = [0] * total_lat
        for e in end:
            iops[int(e - base_t)] += 1

    print_statistics(bw_mbs, iops, lat)


def bench_test(cluster, pool, threads_num, reads_num, use_zipf, zipf_parm):
    ioctx = cluster.open_ioctx(pool)

    objs = prepare_objs(ioctx, reads_num, use_zipf, zipf_parm)

    read_res = []
    for i in range(reads_num):
        read_res.append(dict(start=0.0, end=0.0, len=0, finish=False))

    read_handler = []
    with ThreadPoolExecutor(max_workers=threads_num) as worker:
        for i in range(reads_num):
            read_handler.append(worker.submit(obj_aio_read, i, ioctx, read_res, objs[i]))
    
    wait_finish(read_handler)
    ioctx.close()

    collect_and_print_statistics(read_res)

def main():
    args = parse_args()

    cluster = init(args.conf,args.keyring)

    if cluster.pool_exists(args.pool):
        new_create = False
    else:
        cluster.create_pool(args.pool)
        new_create = True

    threads = args.threads if args.threads < args.total_reads else args.total_reads
    bench_test(cluster, args.pool, threads, 
        args.total_reads, args.enable_zipf_distribution, args.zipf_parm)
    
    if new_create:
        cluster.delete_pool(args.pool)        

def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="ceph read benchmark"
    )

    parser.add_argument(
        "--conf",
        "-c",
        type=str,
        default="/etc/ceph/ceph.conf",
        help="The path of ceph config file.",
    )

    parser.add_argument(
        "--keyring",
        "-k",
        type=str,
        default="/etc/ceph/ceph.client.admin.keyring",
        help="The path of keyring."
    )

    parser.add_argument(
        "--pool",
        "-p",
        type=str,
        default="ec-fast-pool",
        help="The test pool. "
        "It will be created if not exist and delete after test."
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=10,
        choices=range(0,31),
        metavar='NUM_THREADS',
        help="Number of concurrent threats. "
    )

    parser.add_argument(
        "--total-reads",
        "-n",
        type=int,
        default=1000,
        choices=range(1,int(1e4)+1),
        metavar='NUM_READS',
        help="Total number of reads. "
    )

    parser.add_argument(
        "--enable-zipf-distribution",
        "-z",
        action='store_true',
        default=False,
        help="Wheather use zipf distribution. "
        "Otherwise proceed sequantial read"
    )

    parser.add_argument(
        "--zipf-parm",
        "-a",
        type=float,
        default=1.1,
        metavar='ZIPF_PARM',
        help="The parameter of zipf distribution. "
    )

    return parser.parse_args()

if __name__ == '__main__':
    main()
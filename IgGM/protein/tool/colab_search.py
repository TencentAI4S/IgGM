# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os


def get_mmseq_version(mmseqs='mmseqs'):
    version = os.popen(mmseqs).readlines()[5].split(':')[-1].strip()
    return version


class ColabSearch:
    def __init__(
        self,
        base,
        db1_path,
        db2_path=None,
        db3_path=None,
        use_env=False,
        use_templates=False,
        filter=False,
        num_threads=1,
        index=False,
        db_load_mode='0',
        expand_eval='inf',
        align_eval=10,
        diff=3000,
        qsc=-20.0,
        max_accept=1e6,
        mmseqs='mmseqs',
    ):
        self.mmseqs = mmseqs
        self.version = get_mmseq_version(self.mmseqs)
        print('mmseqs version: {}'.format(self.version))
        self.num_threads = num_threads
        self.thread_params = f'--threads {self.num_threads}'
        self.base_dir = base
        os.makedirs(self.base_dir, exist_ok=True)
        self.filter = 1 if filter else 0
        self.use_env = use_env
        self.use_templates = use_templates
        self.expand_eval = expand_eval
        if self.filter:
            align_eval = 10
            qsc = 0.8
            max_accept = 1e5
        self.qsc = qsc
        self.align_eval = align_eval
        self.max_accept = int(max_accept)
        if index:
            SEQ = '.idx'
            ALN = '.idx'
            IDX = '.idx'
            os.environ['MMSEQS_IGNORE_INDEX'] = '0'
        else:
            SEQ = '_seq'
            ALN = '_aln'
            IDX = ''
            os.environ['MMSEQS_IGNORE_INDEX'] = '1'

        os.environ['MMSEQS_CALL_DEPTH'] = '1'
        self.db1_path = db1_path
        self.db1_seq_path = f'{self.db1_path}{SEQ}'
        self.db1_aln_path = f'{self.db1_path}{ALN}'
        self.db1_idx_path = f'{self.db1_path}{IDX}'
        # assert os.path.exists(self.db1_seq_path), f'not exist msa search database: ({self.db1_path})'
        self.db2_path = db2_path
        self.db2_seq_path = f'{self.db2_path}/{SEQ}'
        self.db2_aln_path = f'{self.db2_path}/{ALN}'
        self.db2_idx_path = f'{self.db2_path}/{IDX}'

        self.db3_path = db3_path
        self.db3_seq_path = f'{self.db3_path}/{SEQ}'
        self.db3_aln_path = f'{self.db3_path}/{ALN}'

        self.db_load_mode = db_load_mode
        self.search_param = f'--num-iterations 3 --db-load-mode {self.db_load_mode} -a -s 8 -e 0.1 --max-seqs 10000'
        self.filter_param = (
            f'--filter-msa {self.filter} --filter-min-enable 1000 --diff {diff} '
            f'--qid 0.0,0.2,0.4,0.6,0.8,1.0 --qsc 0 --max-seq-id 0.95'
        )
        self.expand_param = (
            f'--expansion-mode 0 -e {self.expand_eval} --expand-filter-clusters {self.filter} ' f'--max-seq-id 0.95'
        )
        os.makedirs(self.base_dir, exist_ok=True)
        db_load_param = f'--db-load-mode {self.db_load_mode}'
        align_eval_param = f'-e {self.align_eval}'
        max_accept_param = f'--max-accept {self.max_accept}'
        qsc_param = f'--qsc {self.qsc}'

        self.res_dir = f'{self.base_dir}/res'
        self.res_exp_dir = f'{self.base_dir}/res_exp'
        self.qdb_dir = f'{self.base_dir}/qdb'
        self.qdb_h_dir = f'{self.base_dir}/qdb_h'
        self.tmp_dir = f'{self.base_dir}/tmp'
        self.prof_res_dir = f'{self.base_dir}/prof_res'
        self.pdb_res_dir = f'{self.base_dir}/res_pdb'
        self.res_exp_realign_dir = f'{self.base_dir}/res_exp_realign'
        self.res_exp_realign_filter_dir = f'{self.base_dir}/res_exp_realign_filter'
        cmds = [
            f'{self.mmseqs} search {self.qdb_dir} {self.db1_path} {self.res_dir} {self.tmp_dir} {self.search_param}',
            (
                f'{self.mmseqs} expandaln {self.qdb_dir} {self.db1_seq_path} {self.res_dir} {self.db1_aln_path} '
                f'{self.res_exp_dir} {db_load_param} {self.thread_params} {self.expand_param}'
            ),
            f'{self.mmseqs} mvdb {self.base_dir}/tmp/latest/profile_1 {self.prof_res_dir}',
            f'{self.mmseqs} lndb {self.qdb_h_dir} {self.base_dir}/prof_res_h',
            (
                f'{self.mmseqs} align {self.prof_res_dir} {self.db1_seq_path} {self.res_exp_dir} '
                f'{self.res_exp_realign_dir} {db_load_param} {self.thread_params} {align_eval_param} {max_accept_param} '
                f'--alt-ali 10 -a'
            ),
            (
                f'{self.mmseqs} filterresult {self.qdb_dir} {self.db1_seq_path} {self.res_exp_realign_dir} '
                f'{self.res_exp_realign_filter_dir} {db_load_param} {self.thread_params} --qid 0 {qsc_param} --diff 0 '
                f'--max-seq-id 1.0 --filter-min-enable 100'
            ),
            (
                f'{self.mmseqs} result2msa {self.qdb_dir} {self.db1_seq_path} {self.res_exp_realign_filter_dir} '
                f'{self.base_dir}/uniref.a3m --msa-format-mode 6 {db_load_param} {self.thread_params} {self.filter_param}'
            ),
            f'{self.mmseqs} rmdb {self.res_exp_realign_filter_dir}',
            f'{self.mmseqs} rmdb {self.res_exp_realign_dir}',
            f'{self.mmseqs} rmdb {self.res_exp_dir}',
            f'{self.mmseqs} rmdb {self.res_dir}',
        ]
        if self.use_templates:
            # assert os.path.exists(self.db2_path), f'not exist template search database: ({self.db2_path})'
            cmds.extend(
                [
                    (
                        f'{self.mmseqs} search {self.prof_res_dir} {self.db2_path} {self.pdb_res_dir} {self.tmp_dir} '
                        f'{db_load_param} {self.thread_params} -s 7.5 -a -e 0.1'
                    ),
                    (
                        f'{self.mmseqs} convertalis {self.prof_res_dir} {self.db2_idx_path} {self.pdb_res_dir} {self.db2_path}.m8 '
                        f'--format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar '
                        f'{db_load_param} {self.thread_params}'
                    ),
                    f'{self.mmseqs} rmdb {self.pdb_res_dir}',
                ]
            )

        if self.use_env:
            # assert os.path.exists(self.db3_path), f'not exist mmseqs env search database: ({self.db3_path})'
            self.env_res_dir = f'{self.base_dir}/res_env'
            self.env_exp_dir = f'{self.base_dir}/res_env_exp'
            self.env_exp_realgin_dir = f'{self.base_dir}/res_env_exp_realign'
            self.env_exp_realgin_filter_dir = f'{self.base_dir}/res_env_exp_realign_filter'
            cmds.extend(
                [
                    (
                        f'{self.mmseqs} search {self.prof_res_dir} {self.db3_path} {self.env_res_dir} {self.tmp_dir} '
                        f'{self.thread_params} {self.search_param}'
                    ),
                    (
                        f'{self.mmseqs} expandaln {self.prof_res_dir} {self.db3_seq_path} {self.env_res_dir} '
                        f'{self.db3_aln_path} {self.env_exp_dir} -e {self.expand_eval} --expansion-mode 0 {db_load_param} '
                        f'{self.thread_params}'
                    ),
                    (
                        f'{self.mmseqs} align {self.tmp_dir}/latest/profile_1 {self.db3_seq_path} {self.env_exp_dir} '
                        f'{self.env_exp_realgin_dir} {db_load_param} {self.thread_params} {align_eval_param} '
                        f'{max_accept_param} --alt-ali 10 -a'
                    ),
                    (
                        f'{self.mmseqs} filterresult {self.qdb_dir} {self.db3_seq_path} {self.env_exp_realgin_dir} '
                        f'{self.env_exp_realgin_filter_dir} {db_load_param} {self.thread_params} --qid 0 {qsc_param} '
                        f'--diff 0 --max-seq-id 1.0 --filter-min-enable 100'
                    ),
                    (
                        f'{self.mmseqs} result2msa {self.qdb_dir} {self.db3_seq_path} {self.env_exp_realgin_filter_dir} '
                        f'{self.base_dir}/bfd.mgnify30.metaeuk30.smag30.a3m --msa-format-mode 6 {db_load_param} '
                        f'{self.thread_params} {self.filter_param}'
                    ),
                    f'{self.mmseqs} rmdb {self.env_exp_realgin_filter_dir}',
                    f'{self.mmseqs} rmdb {self.env_exp_realgin_dir}',
                    f'{self.mmseqs} rmdb {self.env_exp_dir}',
                    f'{self.mmseqs} rmdb {self.env_res_dir}',
                ]
            )

        cmds.extend(
            [
                f'{self.mmseqs} rmdb {self.qdb_dir}',
                f'{self.mmseqs} rmdb {self.base_dir}/qdb_h',
                f'{self.mmseqs} rmdb {self.base_dir}/res',
            ]
        )

        cmds.extend([f'rm -f -- {self.prof_res_dir}*', f'rm -rf -- {self.tmp_dir}'])
        self.cmds = cmds

    def tsv2exprofiledb(self, src_dir, dst_dir=None):
        dst_dir = dst_dir or src_dir
        os.system(f'{self.mmseqs} tsv2exprofiledb {src_dir} {dst_dir}')

    @classmethod
    def setup_database(cls, database_dir: str):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, query):
        createdb_cmd = f'{self.mmseqs} createdb {query} {self.qdb_dir}'
        cmds = [
            createdb_cmd,
        ] + self.cmds
        for i, cmd in enumerate(cmds):
            print(i, ' th: ', cmd)
            os.system(cmd)

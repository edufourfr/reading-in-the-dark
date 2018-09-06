"""
Discrete logarithm class. Uses baby step giant step. Precomputations are stored
in a postgresql database.
Database credentials are defined here.
"""
import math
import os
import psycopg2
from .utils import exp, is_scalar, Serializable

DATABASE = "discretelogarithm"
USER = "dbuser"
PASSWORD = "password"

assert (
    'PYTHONHASHSEED' in os.environ and os.environ['PYTHONHASHSEED'] == '0'
), (
    """Please disable hash randomization by entering "export PYTHONHASHSEED=0"
    into your shell."""
)


class PreCompBabyStepGiantStep:
    """ Baby Step Giant Step with precomputation of the Giant steps. """

    ext_ = 'bsgs'

    def __init__(
        self,
        groupObj=None,
        base=None,
        minimum=0,
        maximum=0,
        step=0,
    ):
        assert is_scalar(minimum)
        assert is_scalar(maximum)
        assert isinstance(step, int)
        assert not step < 0
        self.group = groupObj
        self.base = base
        self.minimum = math.floor(minimum/step)*(step)
        self.maximum = math.ceil(maximum//step)*(step)
        self.step = step

        self.table = 'dlogs{}'.format(hash(self.base))

        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                ("create table {} (hash bigint,"
                 " log bigint, UNIQUE(hash, log));").format(self.table)
            )
            conn.commit()
        except Exception as e:
            print(e)
            cursor.execute('rollback;')
            conn.commit()

    def precomp(self):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.itersize = 10000
        batch_size = 500
        conn.set_session(autocommit=False)

        # Check if table has elements
        try:
            cursor.execute(
                'insert into {}(hash, log) values (1, 1)'.format(
                    self.table,
                )
            )
            conn.commit()
        except psycopg2.IntegrityError as e:
            if e.pgcode == '23505':
                print(
                    "Database seems to have already been filled (at least",
                    "partially). We do not currently support filling",
                    "databases in multiple sessions. If you believe the",
                    "database is incomplete, please drop the table and run",
                    "precomputation again."
                )
                return
            else:
                raise e

        i = 1
        mult = exp(self.base, self.step)

        total = (self.maximum-self.minimum)//self.step + 1
        preparation = (
            'prepare ins as insert into'
            ' {}(hash, log) values '.format(self.table) +
            ','.join(
                [
                    '(${}, ${})'.format(
                        2*i+1,
                        2*(i+1)
                    ) for i in range(batch_size)
                ]
            ) + ';'
        )
        cursor.execute(preparation)
        z = self.group.random()
        one = z/z
        curr = self.base ** (self.minimum * one)
        for i in range(total//batch_size):
            L = []
            for j in range(batch_size):
                if (i*batch_size+j) % 100000 == 0:
                    print(
                        'Precomputation step {} out of {}.'.format(
                            i*batch_size+j,
                            total
                        )
                    )
                    ratio = (i*batch_size+j) / int(total)
                    print(
                        '|' +
                        '='*(round(40*ratio)) +
                        ' '*(40 - round(40*ratio)) +
                        '| {}%'.format(round(ratio*100))
                    )
                h = hash(curr)
                L.append(h)
                L.append(self.minimum+self.step*(i*batch_size+j))
                curr *= mult
            ins = 'execute ins (' + ','.join([str(x) for x in L]) + ');'
            cursor.execute(ins)
            if (i*batch_size) % 1000000 == 0:
                conn.commit()
        conn.commit()
        k = (total//batch_size) * batch_size
        while k < total+1:
            curr *= mult
            h = hash(curr)
            cursor.execute(
                'insert into {}(hash, log) values ({}, {})'.format(
                    self.table,
                    h,
                    self.minimum+self.step*k
                )
            )
            k += 1
        conn.commit()

    def get_conn(self):
        try:
            return psycopg2.connect(
                dbname=DATABASE,
                user=USER,
                host='localhost',
                password=PASSWORD
            )
        except psycopg2.OperationalError as e:
            conn = psycopg2.connect(
                dbname='postgres',
                user=USER,
                host='localhost',
                password=PASSWORD)
            cursor = conn.cursor()
            cursor.execute("create database discretelogarithm;")
            return self.get_conn()

    def solve(self, target):
        assert (
            type(self.base) == type(target)
        ), (
            'Element and base are not the same type.'
        )
        curr = target
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'prepare sel as select log from '
            '{} where hash=$1'.format(self.table)
        )
        for i in range(0, self.step):
            h = hash(curr)
            cursor.execute("execute sel ({})".format(h))
            line = cursor.fetchone()
            if not line:
                curr *= self.base
                continue
            [first] = line
            if exp(self.base, first) == curr:
                return first - i
            rest = cursor.fetchall()
            for c in rest:
                if exp(self.base, c) == curr:
                    return c - i
            return None
            curr *= self.base
        print('Aborting discrete logarithm.')
        return None

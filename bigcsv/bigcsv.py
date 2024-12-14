import os
import boto3
import pandas as pd
import anndata as an
import pathlib
import warnings
from scipy.sparse import csr_matrix
from typing import Any
import logging
import subprocess
import sys
from tempfile import mkdtemp
from shutil import rmtree
from math import ceil
import csv

warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # ignore the anndata future warning, fairly unncessary


LOGGER = logging.getLogger("bigcsv")
LOGGER.setLevel("INFO")

try:
    import pgzip as gzip
except ImportError:
    import gzip

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x


_open = open


def exopen(csv: str, mode: str = "r", *args, njobs=-1, **kwargs):

    if njobs == -1:
        njobs = os.cpu_count()
    if csv.endswith(".gz"):
        try:
            return gzip.open(
                csv, mode + "t" if not mode.endswith("b") else mode, *args, **kwargs
            )
        except:
            return gzip.open(csv, mode + "t" if not mode.endswith("b") else mode)
    return _open(csv, mode, *args, **kwargs)


def transpose_csv(
    inputfile: str,
    outfile: str,
    chunksize,
    insep=",",
    outsep=",",
) -> None:
    """
    Calculates the transpose of a file too large to fit in memory.

    Parameters:
    inputfile: Path to input file
    outfile: Path to output file (transposed input file)
    chunksize: Number of lines per iteration
    insep: Separator for input file,  default is ','
    outsep: Separator for output file, default is ','

    Returns:
    None
    """

    tmpfolder = mkdtemp()
    chunkfolder = os.path.join(tmpfolder, "chunks")
    os.makedirs(chunkfolder, exist_ok=True)
    tfile = (
        outfile
        if not outfile.endswith(".gz")
        else os.path.join(tmpfolder, os.path.basename(outfile)[:-3])
    )
    if inputfile.endswith(".gz"):
        LOGGER.info("Uncompressing..")
        nfile = os.path.join(tmpfolder, os.path.basename(inputfile)[:-3])
        with exopen(inputfile, "r") as inp, exopen(nfile, "w") as out:
            out.write(inp.read())
        inputfile = nfile
    # First, get the number of lines in the file (total number we have to process)
    with open(inputfile) as f:
        lines = len(f.readlines())
    LOGGER.info(f"Number of lines to process: {lines - 1}")

    num_chunks = ceil(lines / chunksize)
    iterator = enumerate(
        pd.read_csv(
            inputfile,
            header=None,
            index_col=None,
            sep=insep,
            chunksize=chunksize,
            keep_default_na=False,
        )
    )
    chunks = []
    if LOGGER.getEffectiveLevel() >= logging.INFO:
        iterator = tqdm(iterator, total=num_chunks)
    for l, df in iterator:
        df: pd.DataFrame = df.T.fillna("")
        chunks.append(os.path.join(chunkfolder, f"chunk_{l}"))
        with open(chunks[-1], "w") as f:
            for _, row in df.iterrows():
                f.write(outsep.join([str(x) for x in row]) + "\n")

    LOGGER.debug(f"Combining chunks from {chunkfolder} into {tfile}...")

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        cstring = '"' + '" "'.join(chunks) + '"'
        psep_arg = f"-d '{outsep}'" if outsep != "\t" else ""
        os.system(f"paste {psep_arg} {cstring} > {tfile}")
    else:

        cstring = '"' + '", "'.join(chunks) + '"'
        p = subprocess.Popen(
            f'powershell.exe -ExecutionPolicy RemoteSigned paste-content -Path {cstring} -Delimiter "{outsep}" > {tfile}"',
            stdout=sys.stdout,
        )
        p.communicate()

    LOGGER.info("Finished combining chunks, deleting chunks...")
    rmtree(chunkfolder)
    if os.path.dirname(outfile):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if outfile.endswith(".gz"):
        LOGGER.info("Compressing..")
        with open(tfile, "r") as inp, exopen(outfile, "w") as out:
            for line in inp:
                out.write(line)

    LOGGER.info("Done.")
    rmtree(tmpfolder)



def to_h5ad(
    file: str,
    outfile: str,
    sep: str,
    sparsify: bool,
    chunksize: int,
    quiet: bool=False,
    compression: str='infer',
    index_col: str=None,
    dtype: Any=None,
):
    chunkified = pd.read_csv(
        file, 
        chunksize=chunksize, 
        index_col=index_col, 
        compression=compression, 
        dtype=dtype,
        sep=sep,
    )

    with open(file) as f:
        lines = len(f.readlines()) - 1

    anndatas = []
    num_chunks = lines // chunksize + int(lines % chunksize == 0)

    for chunk, data in zip(range(0, num_chunks + 1), chunkified):
        if not quiet: print(f'Working on chunk {chunk}/{num_chunks}')

        if sparsify:
            df = an.AnnData(
                X=csr_matrix(data.values),
            )
            df.var.index = data.columns.values 
            df.obs.index = data.index.values
        else:
            df = an.AnnData(data)

        anndatas.append(df)

    if not quiet: print('Concatenating h5ad\'s')
    df = an.concat(anndatas)
    df.obs = df.obs.reset_index(drop=False)
    
    if not quiet: print('Writing h5ad to file')
    df.write_h5ad(outfile)

class BigCSV:
    def __init__(
        self,
        file: str, 
        outfile: str=None, 
        insep: str=',', 
        outsep: str=',',
        chunksize: str=400, 
        save_chunks: bool=False,
        quiet: bool=False,
    ):
        self.file = file 
        self.outfile = outfile
        self.insep = insep 
        self.outsep = outsep
        self.chunksize = chunksize
        self.save_chunks = save_chunks
        self.quiet = quiet

        outfile_split = file.split('/')
        self.outfile_name = outfile_split[-1][:-4] #takes /path/to/file.csv --> file 


    def transpose_csv(
        self,
        outfile: str=None,
    ):
        if outfile is None and self.outfile is None:
            raise ValueError("Error, either self.outfile must not be None or outfile must not be None.")

        transpose_csv(
            inputfile=self.file, 
            outfile=(outfile if outfile is not None else self.outfile), 
            insep=self.insep, 
            outsep=self.outsep,
            chunksize=self.chunksize, 
            save_chunks=self.save_chunks,
            quiet=self.quiet, 
        )

    def to_h5ad(
        self,
        outfile: str=None,
        sparsify: bool=False,
        compression: str='infer',
        lines: int=None,
        dtype: Any=None,
        index_col: str=None,
        index: bool=True,
    ):
        if pathlib.Path(outfile).suffix != 'h5ad':
            warnings.warn('Suffix of outfile is not .h5ad, although it is being converted to an h5ad.')

        if outfile is None and self.outfile is None:
            raise ValueError("Error, either self.outfile must not be None or outfile must not be None.")

        to_h5ad(
            file=self.file,
            outfile=(outfile if outfile is not None else self.outfile),
            sep=self.insep,
            chunksize=self.chunksize,
            quiet=self.quiet,
            sparsify=sparsify,
            compression=compression,
            dtype=dtype,
            index_col=index_col,
            index=index,
        )

    def upload(
        self, 
        bucket: str,
        endpoint_url: str,
        aws_secret_key_id: str,
        aws_secret_access_key: str,
        remote_file_key: str=None,
        remote_chunk_path: str=None, 
    ) -> None:
        """Uploads the chunks and/or transposed file to the given S3 bucket.

        :param bucket: Bucket name
        :type bucket: str
        :param endpoint_url: S3 endpoint
        :type endpoint_url: str
        :param aws_secret_key_id: AWS secret key for your account
        :type aws_secret_key_id: str
        :param aws_secret_access_key: Specifies the secret key associated with the access key
        :type aws_secret_access_key: str
        :param remote_file_key: key to upload file to in S3. Must be complete path, including file name , defaults to None
        :type remote_file_key: str, optional
        :param remote_chunk_path: Optional, key to upload chunks to in S3. Must be a folder-like path, where the chunks will be labeled as chunk_{outfile_name}_{l}.csv, defaults to None
        :type remote_chunk_path: str, optional
        :raises ValueError: If chunks have been deleted but requested to be uploaded, then we have to error since there is nothing to upload.
        """

        if remote_chunk_path and not self.save_chunks: # if remote_chunk_path is not None and self.save_chunks=False, then we cannot upload!
            raise ValueError('Error, Transpose class not initialized with save_chunks=True, so chunks have been deleted. Rerun with save_chunks=True, or call upload method with remote_chunk_path=None.')
        
        # Defines upload function and uploades combined data after all chunks are generated
        s3 = boto3.resource(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_secret_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        remote_file_key = (self.file if not remote_file_key else remote_file_key)

        if remote_file_key:
            if not self.quiet: print(f'Uploading {self.outfile} transposed to {remote_file_key}')
            s3.Bucket(bucket).upload_file(
                file=self.outfile,
                Key=remote_file_key,
            )

        if remote_chunk_path:
            if not self.quiet and remote_chunk_path: print(f'Uploading chunks to {remote_chunk_path}')
            for file in os.listdir(self.chunkfolder):
                if not self.quiet: print(f'Uploading {file}')
                
                s3.Bucket(bucket).upload_file(
                    file=os.path.join(self.chunkfolder, file),
                    Key=os.path.join(remote_chunk_path, file)
                )

    def __repr__(self) -> str:
        return f"file={self.file}, outfile={self.outfile}, chunksize={self.chunksize}"

    def __str__(self) -> str:
        return self.__repr__()

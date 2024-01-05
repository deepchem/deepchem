import deepchem as dc
import posixpath
import os
import ray
from ray.data import Datasink
from ray.data.block import Block, BlockAccessor
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource.filename_provider import FilenameProvider

from typing import Dict, Any, Iterable, Optional, List
from functools import partial

ray.init(num_cpus=4)


class DiskDatasetFilenameProvider(FilenameProvider):

    def __init__(self):
        self.count = 0

    def get_filename_for_block(self, block: Block, task_index: int,
                               block_index: int) -> str:
        print(' task index is ', task_index, ' block index is ', block_index)
        file_id = f"{task_index}_{block_index}"
        return file_id


class DCDiskDatasink(Datasink):

    def __init__(
        self,
        path: str = None,
        try_create_dir: bool = True,
    ):
        """Initialize this datasink.

        Args:
            path: The folder to write files to.
            filesystem: The filesystem to write files to. If not provided, the
                filesystem is inferred from the path.
            try_create_dir: Whether to create the directory to write files to.
            open_stream_args: Arguments to pass to ``filesystem.open_output_stream``.
            filename_provider: A :class:`ray.data.datasource.FilenameProvider` that
                generates filenames for each row or block.
            dataset_uuid: The UUID of the dataset being written. If specified, it's
                included in the filename.
            file_format: The file extension. If specified, files are written with this
                extension.
        """

        self.filename_provider = DiskDatasetFilenameProvider()
        self.path = 'data'
        self.try_create_dir = try_create_dir
        self.has_created_dir = False

    def on_write_start(self):
        os.makedirs(self.path, exist_ok=True)

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> Any:
        num_rows_written = 0
        shard_id = 0
        block_index = 0
        for block in blocks:
            block = BlockAccessor.for_block(block)
            if block.num_rows() == 0:
                continue

            self.write_block(block, block_index, ctx)
            shard_id += 1
            num_rows_written += block.num_rows()
            block_index += 1

        return True

    def write_block(self, block: BlockAccessor, block_index: int,
                    ctx: TaskContext):
        filename = self.filename_provider.get_filename_for_block(
            block, ctx.task_idx, block_index)
        write_path = posixpath.join(self.path, filename)

        print('number of rows in block ', block.num_rows(), ' block index ',
              block_index, ' task index ', ctx.task_idx)
        with open(filename, 'w') as f:
            f.write(f'{str(block_index)}-{str(ctx.task_idx)}')

    def on_write_complete(self, write_results: List[Any]) -> None:
        pass


datasink = DCDiskDatasink()


def featurize(row: Dict[str, Any],
              featurizer,
              x='smiles',
              y='logp') -> Dict[str, Any]:
    # row[x + '_features'] = featurizer(row['smiles'])
    return row


featurize_batches = partial(featurize, featurizer=dc.feat.CircularFingerprint())

ds = ray.data.read_csv('zinc1k.csv').map_batches(featurize_batches, num_cpus=4)

print('num blocks is ', ds.num_blocks())

ds.write_datasink(datasink)

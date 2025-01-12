import os
import sys
import uuid
import time
import datetime as dt
from pathlib import Path
from numpy import add
import pandas as pd
import xlwings as xw

def getTimestamp(ts: dt.datetime) -> str:
    return ts.strftime('%Y-%m-%d_%H-%M-%S')


if __name__ == '__main__':
    #region Path examples

    # Get the current working directory
    cwd = Path.cwd()  # cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # Get the home directory
    home = Path.home()
    print(f"Home directory: {home}")

    #endregion Path examples

    backupKind = 'Full'  # Full, Incremental
    backupKind = 'Incremental'
    incremental = backupKind == 'Incremental'

    pathRoot = Path(r'C:/AddedValues Dropbox/Fælles')
    dfDirs = pd.DataFrame(columns=['Path', 'TotalSize', 'FileCount', 'FolderCount', 'Modified', 'mtime', 'ctime'])

    if incremental:
        # Get the latest file with the directory and file statistics.
        pathsAllPrevDirStats = ([path for path in Path('.').glob('DirStats*.xlsx')])
        pathsAllPrevDirStats.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        if len(pathsAllPrevDirStats) == 0:
            raise FileNotFoundError('No file with directory statistics found, when running incremental backup')

        pathPrevDirStats = pathsAllPrevDirStats[0]
        wb = xw.Book(pathPrevDirStats, visible=False)
        ws = wb.sheets[0]
        dfDirsPrev = ws.range('A10').options(pd.DataFrame, index=False, expand='table').value
        wb.close()
        print(f'Loaded {len(dfDirs)} rows from {pathPrevDirStats}')

        # Convert to dictionary for faster lookup: Key = Path, Value = mtime (modified time as seconds since epoch).
        dictPrev = dict(zip(dfDirsPrev['Path'], dfDirsPrev['mtime']))


    # Walk through the directory tree.
    tbegin = time.time()
    nFolderTot = 0
    nFolder = 0
    nFileTot = 0
    rows = []
    for root, dirs, files in pathRoot.walk(top_down=True):

        nFile = 0
        totalSize = 0
        for file in files:
            pathFile = str(root / file)
            stat = (root / file).stat()
            tmodif = stat.st_mtime      # Modified at seconds since epoch (01-01-1970)
            tbirth = stat.st_birthtime  # Created at seconds since epoch (01-01-1970)

            addFile = not incremental
            if incremental:
                # Criteria: Check if the file has been modified since the last backup.
                mTimePrev = dictPrev.get(str(pathFile), None)
                addFile   = (mTimePrev is None or mTimePrev < tmodif)

            if addFile:
                # if (tmodif < tbirth):
                #     print(f'Warning: {file} has modified time less than created time')
                dtModified = dt.datetime.fromtimestamp(tmodif)

                # FolderCount = -1 indicates that this row represents a file.
                row = {'Path': str(root / file), 'TotalSize': (root / file).stat().st_size, 'FileCount': 1, 'FolderCount': -1, \
                        'Modified': dtModified, 'mtime': tmodif, 'ctime': tbirth}
                rows.append(row)
                nFile += 1
                totalSize += stat.st_size
                print(f'File: {file}, Modified/Added/Renamed: {dtModified}')

        nFileTot += nFile   
        
        # Include the current folder, if any modified files were found or the folder is new and empty.
        stat = root.stat()
        tmodif = stat.st_mtime      # Modified at seconds since epoch (01-01-1970)
        tbirth = stat.st_birthtime  # Created at seconds since epoch (01-01-1970)
        addFolder = not incremental
        if incremental:
            # Criteria: Check if the folder has been modified since the last backup.
            mTimePrev = dictPrev.get(str(root), None)
            addFolder = (mTimePrev is None or mTimePrev < tmodif)

        if addFolder:
            dtModified = dt.datetime.fromtimestamp(tmodif)
            row = {'Path': str(root), 'TotalSize': totalSize, 'FileCount': len(files), 'FolderCount': len(dirs), \
                    'Modified': dtModified, 'mtime': tmodif, 'ctime': tbirth}
            rows.append(row)
            nFolder += 1

        nFolderTot += 1
        if nFolderTot % 1_000 == 0:
            print(f'Folders done: {nFolderTot}')
        if nFolderTot > 1:
            break
        pass

    tend = time.time()
    print(f'Time elapsed: {tend - tbegin:.2f} seconds')
    dfDirs = pd.DataFrame(rows)
    dfDirs['UUID']  = dfDirs['Path'].apply(lambda x: str(uuid.uuid4()))
    dfDirs['LenPath'] = dfDirs['Path'].apply(lambda x: len(x))

    # # Sort the rows by the path.
    # dfDirs.sort_values(by=['Path'], ascending=[True], inplace=True)

    # Reorder the columns so that 'Path' is the last column.
    cols = dfDirs.columns[1:].tolist()
    cols.append('Path')
    dfDirs = dfDirs[cols]
    print(f'Number of folders={nFolder} and files = {nFileTot}')
    
    pathDirStats = f'DirStats_{backupKind}_{getTimestamp(dt.datetime.now())}.xlsx'

    if False:
        dfDirs.to_excel(pathDirStats, index=False)
    else:
        wb = xw.Book(visible=False)
        ws = wb.sheets[0]
        ws.range('A10').options(index=False).value = dfDirs
        ws.range('A1').value = ['Backup', backupKind]
        ws.range('A2').value = ['Root', str(pathRoot)]
        ws.range('A3').value = ['Performed', dt.datetime.now()]
        ws.range('A4').value = ['Updated folders', nFolder]
        ws.range('A5').value = ['Updated files', nFileTot]
        wb.save(pathDirStats)
        wb.close()
        
    pass
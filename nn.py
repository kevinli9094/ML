from math import tanh
from sqlite3 import dbapi2 as sqlite


def dtanh(y):
    return 1.0 - y * y

class searchnet:
    def __init__(self, dbname):
        self.con = sqlite.connect(dbname)

    def __del__(self):
        self.con.close()

    def makeTable(self):
        self.con.execute('create table hiddennode(create_key)')
        self.con.execute('create table wordhidden(fromid, toid, strength)')
        self.con.execute('create table urlhidden(fromid, toid, strength)')
        self.con.commit()

    def getStrength(self, fromid, toid, layer):
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'urlhidden'
        res = self.con.execute('select strength from %s where fromid=%d and toid=%d' % (table, fromid, toid)).fetchone()
        if res is None:
            if layer == 0 : return -0.2
            else : return 0
        return res[0]

    def setStrength(self, fromid, toid, layer, strength):
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'urlhidden'
        res = self.con.execute('select rowid from  %s where fromid=%d and toid=%d' % (table, fromid, toid)).fetchone()
        if res is None:
            self.con.execute('insert into %s (fromid, toid, strength) values (%d, %d, %d)' % (table, fromid, toid, strength))
        else:
            rowid = res[0]
            self.con.execute('update %s set strength = %d where rowid=%d' % (table, strength, rowid))

    def generatehiddennode(self, wordids, urlids):
        if len(wordids) > 3 : return None
        createkey = '_'.join(sorted([str[wi] for wi in wordids]))
        res = self.con.execute('select row id from hiddennode where create_key=%d' % createkey).fetchone()

        if res is None :
            cur = self.con.execute('insert into hiddennode (create_key) values (%s)' % createkey)
            hiddennodeId = cur.lastrowid
            for (wId) in wordids:
                self.setStrength(wId, hiddennodeId, 0, 1.0/len(wordids))
            for (urlId) in urlids:
                self.setStrength(hiddennodeId, urlId, 1 , 0.1)
            self.con.commit()

    def getAllHiddenIds(self, wordids, urlids):
        list={}
        for (wordId) in wordids:
            cur = self.con.execute('select toid from wordhidden where fromid=%d' % (wordId))
            for row in cur:
                list[row[0]] = 1
        for urlId in urlids:
            cur = self.con.execute('select fromid from urlhidden where toid=%d' % urlId)
            for row in cur:
                list[row[0]] = 1
        return list.keys()

    def setupNetwork(self, wordids, urlids):
        # initial values
        self.wordids = wordids
        self.urlids = urlids
        self.hiddenids = self.getAllHiddenIds(wordids, urlids)

        #node output
        self.ai = [1.0] * len(self.wordIds)
        self.ao = [1.0] * len(self.urlIds)
        self.ah = [1.0] * len(self.hiddenIds)

        #weight matrix
        self.wi = [self.getStrength(wordId, hiddenId, 0)
                   for wordId in self.wordIds
                   for hiddenId in self.hiddenIds]
        self.wo = [self.getStrength(hiddenId, urlId, 1)
                   for hiddenId in self.hiddenIds
                   for urlId in self.urlIds]

    def feedForward(self):
        for i in range(len(self.wordIds)):
            self.ai[i] = 1.0;

        for j in range(len(self.hiddenIds)):
            sum = 0.0
            for i in range(len(self.wordIds)):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        for k in range(len(self.urlIds)):
            sum = 0.0
            for j in range(len(self.hiddenIds)):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def getResult(self, wordids, urlids):
        self.setupNetwork(wordids, urlids)
        return self.feedForward()



    def backPropagation(self, targets, learningRate =0.5):
        outputDelta = [0.0] * len(self.urlIds) # % to change
        for k in range(len(self.urlIds)):
            error = targets[k] - self.ao[k]
            outputDelta[k] = dtanh(self.ao[k]) * error

        hiddenDelta = [0.0] * len(self.hiddenIds)
        for j in range(len(self.hiddenIds)):
            error = 0.0
            for k in range(len(self.urlIds)):
                error += outputDelta[k] * self.wo[j][k]
            hiddenDelta[j] = dtanh(self.ah[j]) * error
        for j in range(len(self.hiddenIds)):
            for k in range(len(self.urlIds)):
                change = outputDelta[k] * self.ah[j]
                self.wo[j][k] += learningRate*change
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hiddenDelta[j] * self.ai[i]
                self.wi[i][j] += learningRate * change

    
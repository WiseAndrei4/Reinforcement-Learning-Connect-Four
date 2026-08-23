class GameDriver:
    #step=x -> randul jucatorului x(1 sau 2)
    def __init__(self):
        self.step=1
        self.board = [[0] * 7 for _ in range(6)]
        self.winner=-1
        self.remaining=6*7
    def process(self, lin, col):
        aux=1
        l=lin-1
        while(l>=0 and self.board[l][col]==self.board[lin][col]):
            aux+=1
            l-=1
        l=lin+1
        while(l<6 and self.board[l][col]==self.board[lin][col]):
            aux+=1
            l+=1
        if aux>=4:
            self.winner=self.step
        aux=1
        c=col-1
        while(c>=0 and self.board[lin][c]==self.board[lin][col]):
            aux+=1
            c-=1
        c=col+1
        while(c<7 and self.board[lin][c]==self.board[lin][col]):
            aux+=1
            c+=1
        if aux>=4:
            self.winner=self.step
        l=lin+1
        c=col+1
        aux=1
        while(l<6 and c<7 and self.board[lin][col]==self.board[l][c]):
            l+=1
            c+=1
            aux+=1
        c=col-1
        l=lin-1
        while(l>=0 and c>=0 and self.board[l][c]==self.board[lin][col]):
            l-=1
            c-=1
            aux+=1
        if aux>=4:
            self.winner=self.step
        aux=1
        l=lin+1
        c=col-1
        while(l<6 and c>=0 and self.board[l][c]==self.board[lin][col]):
            l+=1
            c-=1
            aux+=1
        l=lin-1
        c=col+1
        while(l>=0 and c<7 and self.board[l][c]==self.board[lin][col]):
            l-=1
            c+=1
            aux+=1
        if aux>=4:
            self.winner=self.step
        if self.remaining==0 and self.winner==-1:
            self.winner=0
        return self.winner
    def play(self, col):
        for i in range(5,-1,-1):
            if self.board[i][col]==0:
                self.board[i][col]=self.step
                self.remaining-=1
                if self.process(i, col)!=-1:
                   return self.winner
                self.step+=1
                if self.step==3:
                    self.step=1
                return 'bun'
        return 'gresit'
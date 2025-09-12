import heapq

class Merge:
    def __init__(self, pos: int, rank: int, new_id: int):
        self.pos = pos
        self.rank = rank
        self.new_id = new_id
    
    def __lt__(self, other: "Merge"):
        if self.rank != other.rank:
            return self.rank < other.rank
        else:
            return self.pos < other.pos


class Symbol:
    def __init__(self, c: int, prev: int, nxt: int, len: int):
        # the current symbol ID
        self.c: int = c
        # the prev symbol index
        self.prev = prev
        # the next symbol index
        self.next = nxt
        # the current symbol length
        self.len = len

    def merge_with(self, other: "Symbol", new_c: int):
        """Merge this symbol with another symbol.

        Args:
            other (Symbol): The symbol to merge with.
            new_c (int): The ID of the new merged symbol.
        """
        self.c = new_c
        self.len += other.len
        self.next = other.next

class Word:
    def __init__(self):
        self.symbols: list[Symbol] = []
    
    
    def add(self, c: int, byte_len: int):
        size = len(self.symbols)

        prev, next = -1, -1
        if size > 0:
            prev = size - 1
            self.symbols[prev].next = size

        self.symbols.append(Symbol(c, prev, next, byte_len))
    
    def merge(self, c1: int, c2: int, replacement: int) -> list[tuple[int, int], int]:
        """Merge two symbols into a new symbol.

        Args:
            c1 (int): The first symbol's ID.
            c2 (int): The second symbol's ID.
            replacement (int): The new symbol's ID.

        Returns:
            list[tuple[int, int], int]: A list of tuples containing the changes of pairs and their counts.
        """
        changes: list[tuple[tuple[int, int], int]] = []
        for i, symbol in enumerate(self.symbols):
            if symbol.c == c1 and i + 1 < len(self.symbols) and self.symbols[i + 1].c == c2:
                first = self.symbols[i]
                second = self.symbols[i + 1]

                new_s = Symbol(replacement, first.prev, second.next, first.len + second.len)
                changes.append(((c1, c2), -1))

                if i > 0:
                    prev = self.symbols[i - 1]
                    changes.append(((prev.c, c1), -1))
                    changes.append(((prev.c, replacement), 1))

                self.symbols.insert(i, new_s)
                self.symbols.pop(i + 1)
                self.symbols.pop(i + 1)

                if i + 1 < len(self.symbols):
                    nxt = self.symbols[i + 1]
                    changes.append(((c2, nxt.c), -1))
                    changes.append(((replacement, nxt.c), 1))
        return changes

    def merge_all(self, merge_map: dict[tuple[int, int], tuple[int, int]]):
        """Merge all specified symbol pairs in the word.

        Args:
            merge_map (dict[tuple[int, int], tuple[int, int]]): A mapping of symbol pairs to their merge information. 
                pair -> (merge_order, token_id)
        """
        pq = []
        for i in range(len(self.symbols) - 1):
            pair = (self.symbols[i].c, self.symbols[i + 1].c)
            if pair in merge_map:
                rank, new_id = merge_map[pair]
                heapq.heappush(pq, Merge(i, rank, new_id))

        while pq:
            top = heapq.heappop(pq)

            # skip if the symbol is already merged
            if self.symbols[top.pos].len == 0:
                continue
                
            # skip if it's last symbol of the word
            if self.symbols[top.pos].next == -1:
                continue

            # merge with
            next_pos = self.symbols[top.pos].next
            right = self.symbols[next_pos]

            # check expired queue entry
            # an entry is considered expired if the new_id has changed
            target_new_pair = (self.symbols[top.pos].c, right.c)
            if target_new_pair not in merge_map or merge_map[target_new_pair][1] != top.new_id:
                continue

            self.symbols[top.pos].merge_with(right, top.new_id)
            # Mark the right symbol as removed
            self.symbols[next_pos].len = 0

            # update the prev pointer for the next symbol of the right symbol
            if right.next > -1 and right.next < len(self.symbols):
                self.symbols[right.next].prev = top.pos
            
            # insert the new pair with the previous symbol
            current = self.symbols[top.pos]
            if current.prev >= 0:
                prev = current.prev
                prev_symbol = self.symbols[prev]
                new_pair = (prev_symbol.c, current.c)
                if new_pair in merge_map:
                    rank, new_id = merge_map[new_pair]
                    heapq.heappush(pq, Merge(prev, rank, new_id))

            # insert the new pair with the next symbol
            next = current.next
            if next < len(self.symbols):
                next_symbol = self.symbols[next]
                new_pair = (current.c, next_symbol.c)
                if new_pair in merge_map:
                    rank, new_id = merge_map[new_pair]
                    heapq.heappush(pq, Merge(top.pos, rank, new_id))


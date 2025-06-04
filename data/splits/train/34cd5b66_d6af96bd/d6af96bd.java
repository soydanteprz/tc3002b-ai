import java.io.*;
import java.util.*;
import java.util.logging.LoggingPermission;

public class C_binarytable {

    public static void main(String[] args) throws Exception {

        FastScanner sc = new FastScanner();
        FastWriter writer = new FastWriter();
        int t = sc.nextInt();
        while(t-- > 0) {
            int n = sc.nextInt(), m = sc.nextInt();
            List<Operation> ops = new ArrayList<>();
            char s[][] = new char[n][m];
            for(int i = 0; i < n; i++)
                s[i] = sc.next().toCharArray();

            // diag
            for(int i = 0; i < n - 1; i++)
                for(int j = 0; j < m; j++) {
                    if(s[i][j] == '0')
                        continue;
                    // right diag
                    if(j + 1 < m && s[i+1][j+1] == '1') {
                        s[i][j] = '0';
                        Operation op = new Operation();
                        op.pairs.add(new Pair(i, j));
                        op.pairs.add(new Pair(i + 1, j));
                        op.pairs.add(new Pair(i, j + 1));
                        ops.add(op);

                        op = new Operation();
                        op.pairs.add(new Pair(i, j + 1));
                        op.pairs.add(new Pair(i + 1, j));
                        op.pairs.add(new Pair(i + 1, j + 1));
                        ops.add(op);
                        s[i+1][j+1] = '0';

                        // left diag
                    } else if(j > 0 && s[i+1][j-1] == '1') {
                        s[i][j] = '0';
                        Operation op = new Operation();
                        op.pairs.add(new Pair(i, j));
                        op.pairs.add(new Pair(i, j - 1));
                        op.pairs.add(new Pair(i + 1, j));
                        ops.add(op);

                        op = new Operation();
                        op.pairs.add(new Pair(i, j - 1));
                        op.pairs.add(new Pair(i + 1, j - 1));
                        op.pairs.add(new Pair(i + 1, j));
                        ops.add(op);
                        s[i+1][j-1] = '0';
                    }
                }

            // same row adjacent
            for(int i = 0; i < n; i++)
                for(int j = 0; j < m - 1; j++) {
                    if(s[i][j] == '0')
                        continue;

                    Pair cur = new Pair(i, j);
                    if(i + 1 < n && s[i][j+1] == '1') {
                        s[i][j] = '0';
                        ops.add(get(cur, cur));
                        ops.add(get(cur, new Pair(i, j+1)));
                        s[i][j+1] = '0';

                    } else if(i + 1 == n && s[i][j+1] == '1') {
                        s[i][j] = '0';
                        Pair upper = new Pair(i-1, j);
                        ops.add(get(upper, cur));
                        ops.add(get(upper, new Pair(i, j+1)));
                        s[i][j+1] = '0';
                    }
                }

            // same col adjacent
            for(int i = 0; i < n - 1; i++)
                for(int j = 0; j < m; j++) {
                    if(s[i][j] == '0')
                        continue;
                    Pair prevCol = new Pair(i, j - 1);
                    Pair cur = new Pair(i, j);

                    if(j > 0 && s[i+1][j] == '1') {
                        s[i][j] = '0';
                        ops.add(get(prevCol, cur));
                        ops.add(get(prevCol, new Pair(i+1, j)));
                        s[i+1][j] = '0';

                    } else if(j == 0 && s[i+1][j] == '1') {
                        s[i][j] = '0';
                        ops.add(get(cur, cur));
                        ops.add(get(cur, new Pair(i + 1, j)));
                        s[i+1][j] = '0';
                    }
                }

            // single 1
            for(int i = 0; i < n; i++)
                for(int j = 0; j < m; j++) {
                    if(s[i][j] == '0')
                        continue;
                    Pair prevCol = new Pair(i, j - 1);
                    Pair cur = new Pair(i, j);
                    s[i][j] = '0';

                    if(i < n - 1) {
                        if(j < m-1) {
                            Pair next = new Pair(i, j+1);
                            ops.add(get(cur, next));
                            ops.add(get(cur, new Pair(i+1, j+1)));
                            ops.add(get(cur, new Pair(i+1, j)));

                        } else if(j == m-1) {

                            ops.add(get(prevCol, prevCol));
                            ops.add(get(prevCol, new Pair(i+1, j - 1)));
                            ops.add(get(prevCol, new Pair(i+1, j)));

                        }
                    } else if(i == n-1) {
                        if(j < m-1) {

                            Pair upper = new Pair(i-1, j);
                            ops.add(get(upper, new Pair(i-1, j+1)));
                            ops.add(get(upper, new Pair(i, j+1)));
                            ops.add(get(upper, upper));

                        } else if(j == m-1) {

                            Pair upper = new Pair(i-1, j-1);
                            ops.add(get(upper, upper));
                            ops.add(get(upper, new Pair(i, j-1)));
                            ops.add(get(upper, new Pair(i-1, j)));
                        }
                    }
                }

            int size = ops.size();
            if(size > 3 * n * m)
                throw new RuntimeException();

            writer.println(ops.size());
            for(Operation op : ops)
                writer.println(op.toString());
        }
        writer.close();
    }

    static Operation get(Pair topLeft, Pair ex) {
        Operation op = new Operation();
        for(int i = 0; i <= 1; i++)
            for(int j = 0; j <= 1; j++) {
                Pair p = new Pair(topLeft.x + i, topLeft.y + j);
                if(!p.equals(ex))
                    op.pairs.add(p);
            }
        return op;
    }

    static class Operation {
        List<Pair> pairs = new ArrayList<>();

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for(Pair p : pairs)
                sb.append(p.x+1 + " " + (p.y+1) + " ");
            return sb.toString();
        }
    }

    static final Random random = new Random();

    

    

    

    

    

    
    
    
    static 
    static 
    
    
    

    

    

    static void println(int[] A) {
        for(int e: A) System.out.print(e + " ");
        System.out.println();
    }

    static void println(long[] A) {
        for(long e: A) System.out.print(e + " ");
        System.out.println();
    }

    static void println(List arr) {
        for(Object e: arr) System.out.print(e + " ");
        System.out.println();
    }

    static <T> void println(T A[]) {

        for(T i : A)
            System.out.print(i + " ");
        System.out.println();
    }

    
    static void println(long n) {
        System.out.println(n);
    }

    static void println(String s) {
        System.out.println(s);
    }

    

    

    

    

    
    
    
    
    
    
    
    
    
    
    
    

    private static class Pair implements Comparable<Pair> {
        int x, y;
        Pair(int x, int y) { this.x = x; this.y = y; }
        Pair() {}

        

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return x == pair.x &&
                    y == pair.y;
        }

        

        @Override
        public String toString() {
            return String.format("(%d, %d)", x, y);
        }
    }

    

    

    

    
}

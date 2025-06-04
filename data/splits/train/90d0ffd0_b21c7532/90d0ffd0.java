import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

public class C {

    public static void main(String[] args) {
        FastReader scan = new FastReader();
        PrintWriter out = new PrintWriter(System.out);
        Task solver = new Task();
        int t = scan.nextInt();
        for(int tt = 1; tt <= t; tt++) solver.solve(tt, scan, out);
        out.close();
    }

    static class Task {
        int n, m;
        char[][] a;
        ArrayList<int[]> moves;

        public void solve(int testNumber, FastReader scan, PrintWriter out) {
            n = scan.nextInt();
            m = scan.nextInt();
            a = new char[n][m];
            moves = new ArrayList<>();
            for(int i = 0; i < n; i++) a[i] = scan.next().toCharArray();
            if(n % 2 == 0) {
                for (int i = 0; i < n; i += 2) {
                    if (m % 2 == 0) {
                        for (int j = 0; j < m; j += 2) move(i, j);
                    } else {
                        for (int j = 0; j < m - 3; j += 2) move(i, j);
                        if (one(i, m - 2).size() == 4) {
                            move(i, m - 3);
                            move(i, m - 2);
                        } else {
                            move(i, m - 3);
                            move(i, m - 2);
                        }
                    }
                }
            }
            else {
                for (int i = 0; i < n - 1; i += 2) {
                    if (m % 2 == 0) {
                        for (int j = 0; j < m; j += 2) move(i, j);
                    } else {
                        for (int j = 0; j < m - 3; j += 2) move(i, j);
                        if (one(i, m - 2).size() == 4) {
                            move(i, m - 3);
                            move(i, m - 2);
                        } else {
                            move(i, m - 3);
                            move(i, m - 2);
                        }
                    }
                }
                if (m % 2 == 0) {
                    for (int j = 0; j < m; j += 2) move(n - 2, j);
                } else {
                    for (int j = 0; j < m - 3; j += 2) move(n - 2, j);
                    if (one(n - 2, m - 2).size() == 4) {
                        move(n - 2, m - 3);
                        move(n - 2, m - 2);
                    } else {
                        move(n - 2, m - 3);
                        move(n - 2, m - 2);
                    }
                }
            }
            out.println(moves.size() / 3);
            for(int i = 0; i < moves.size(); i += 3) {
                for(int j = 0; j < 3; j++) out.printf("%d %d ", moves.get(i + j)[0], moves.get(i + j)[1]);
                out.println();
            }
        }

        void move(int y, int x) {
            ArrayList<int[]> o = one(y, x), z = zero(y, x);
            while(!o.isEmpty()) {
                int first = -1, second = -1;
                if(o.size() <= 2) {
                    first = 1;
                    second = 2;
                }
                else {
                    first = 3;
                    second = 0;
                }
                for(int i = 0; i < first; i++) {
                    int[] t = o.get(i);
                    a[t[0] - 1][t[1] - 1] = '0';
                    moves.add(t);
                }
                for(int i = 0; i < second; i++) {
                    int[] t = z.get(i);
                    a[t[0] - 1][t[1] - 1] = '1';
                    moves.add(t);
                }
                o = one(y, x);
                z = zero(y, x);
            }
        }

        ArrayList<int[]> one(int y, int x) {
            ArrayList<int[]> res = new ArrayList<>();
            for(int i = y + 1; i >= y; i--) {
                for(int j = x + 1; j >= x; j--) {
                    if(a[i][j] == '1') res.add(new int[] {i + 1, j + 1});
                }
            }
            return res;
        }
        ArrayList<int[]> zero(int y, int x) {
            ArrayList<int[]> res = new ArrayList<>();
            for(int i = y; i <= y + 1; i++) {
                for(int j = x; j <= x + 1; j++) {
                    if(a[i][j] == '0') res.add(new int[] {i + 1, j + 1});
                }
            }
            return res;
        }
    }

    

    

    

}

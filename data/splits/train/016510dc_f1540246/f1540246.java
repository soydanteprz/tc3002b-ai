import java.io.*;
import java.util.*;

public class C {
    static InputReader in;
    static PrintWriter out;

    public static void main(String[] args) {
        //initReaderPrinter(true);
        initReaderPrinter(false);
        solve(in.nextInt());
        //solve(1);        
    }

    static void solve(int testCnt) {
        for (int testNumber = 0; testNumber < testCnt; testNumber++) {
            int n = in.nextInt(), m = in.nextInt();
            char[][] g = new char[n][];
            for(int i = 0; i < n; i++) {
                g[i] = in.next().toCharArray();
            }
            List<List<Integer>> list = new ArrayList<>();
            for(int i = 0; i < n - 1; i++) {
                for(int j = 0; j < m - 1; j++) {
                    list.addAll(change(g, i, j));
                }
            }
            out.println(list.size());
            for(int i = 0; i < list.size(); i++) {
                for(int j = 0; j < 6; j++) {
                    out.print(list.get(i).get(j) + " ");
                }
                out.println();
            }
        }
        out.close();
    }

    static List<List<Integer>> change(char[][] g, int i, int j) {
        List<List<Integer>> list = new ArrayList<>();
        int cnt = 0;
        cnt += g[i][j] - '0';
        cnt += g[i][j + 1] - '0';
        cnt += g[i + 1][j] - '0';
        cnt += g[i + 1][j + 1] - '0';
        if(cnt != 0) {
            if(cnt == 4) {
                List<Integer> op = new ArrayList<>();
                op.add(i + 1); op.add(j + 1 + 1); op.add(i + 1 + 1); op.add(j + 1); op.add(i + 1 + 1); op.add(j + 1 + 1);
                list.add(op);
                g[i][j + 1] = '0';
                g[i + 1][j] = '0';
                g[i + 1][j + 1] = '0';
                cnt = 1;
            }
            if(cnt == 1) {
                List<Integer> op = new ArrayList<>();
                int zeroCnt = 0;
                for(int k1 = 0; k1 < 2; k1++) {
                    for(int k2 = 0; k2 < 2; k2++) {
                        if(g[i + k1][j + k2] == '0' && zeroCnt < 2) {
                            op.add(i + k1 + 1);
                            op.add(j + k2 + 1);
                            zeroCnt++;
                        }
                        else if(g[i + k1][j + k2] == '1'){
                            op.add(i + k1 + 1);
                            op.add(j + k2 + 1);
                        }
                    }
                }
                list.add(op);
                zeroCnt = 0;
                for(int k1 = 0; k1 < 2; k1++) {
                    for(int k2 = 0; k2 < 2; k2++) {
                        if(g[i + k1][j + k2] == '0' && zeroCnt < 2) {
                            g[i + k1][j + k2] = '1';
                            zeroCnt++;
                        }
                        else if(g[i + k1][j + k2] == '1'){
                            g[i + k1][j + k2] = '0';
                        }
                    }
                }
                cnt = 2;
            }
            if(cnt == 2) {
                List<Integer> op = new ArrayList<>();
                int oneCnt = 0;
                for(int k1 = 0; k1 < 2; k1++) {
                    for(int k2 = 0; k2 < 2; k2++) {
                        if(g[i + k1][j + k2] == '0') {
                            op.add(i + k1 + 1);
                            op.add(j + k2 + 1);
                        }
                        else if(oneCnt < 1){
                            op.add(i + k1 + 1);
                            op.add(j + k2 + 1);
                            oneCnt++;
                        }
                    }
                }
                list.add(op);
                oneCnt = 0;
                for(int k1 = 0; k1 < 2; k1++) {
                    for(int k2 = 0; k2 < 2; k2++) {
                        if(g[i + k1][j + k2] == '0') {
                            g[i + k1][j + k2] = '1';
                        }
                        else if(oneCnt < 1){
                            g[i + k1][j + k2] = '0';
                            oneCnt++;
                        }
                    }
                }
                cnt = 3;
            }
            if(cnt == 3) {
                List<Integer> op = new ArrayList<>();
                for(int k1 = 0; k1 < 2; k1++) {
                    for(int k2 = 0; k2 < 2; k2++) {
                        if(g[i + k1][j + k2] == '1') {
                            op.add(i + k1 + 1);
                            op.add(j + k2 + 1);
                        }
                    }
                }
                list.add(op);
            }
            for(int k1 = 0; k1 < 2; k1++) {
                for(int k2 = 0; k2 < 2; k2++) {
                    g[i + k1][j + k2] = '0';
                }
            }
        }
        return list;
    }

    static void initReaderPrinter(boolean test) {
        if (test) {
            try {
                in = new InputReader(new FileInputStream("src/input.in"));
                out = new PrintWriter(new FileOutputStream("src/output.out"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            in = new InputReader(System.in);
            out = new PrintWriter(System.out);
        }
    }

    
}

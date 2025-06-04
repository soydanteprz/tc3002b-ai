import java.io.*;
import java.sql.Array;
import java.util.*;

public class P1551C {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Task solver = new Task();
        solver.solve(in, out);
        out.close();
    }

    static class Task {
        public void solve(InputReader in, PrintWriter out) {
            int t = in.nextInt();
            while (t-- > 0) {
                int n = in.nextInt();
                String[] s = new String[n];
                int[][] freq = new int[n][6];
                for (int i = 0; i < n; i++) {
                    s[i] = in.nextLine();
                    for (char c : s[i].toCharArray()) {
                        freq[i][c - 'a']++;
                        freq[i][5]++;
                    }
                }
                int totmax = 0;
                for (int i = 0; i < 5; i++) {
                    List<Integer> diffs = new ArrayList<>(n);
                    for (int j = 0; j < n; j++) {
                        diffs.add(2 * freq[j][i] - freq[j][5]);
                    }
                    Collections.sort(diffs);
                    Collections.reverse(diffs);
                    int curr = 0, cnt = 0;
                    while (cnt < diffs.size() && curr + diffs.get(cnt) > 0) {
                        curr += diffs.get(cnt++);
                    }
                    totmax = Math.max(totmax, cnt);
                }
                out.println(totmax);
            }
        }
    }

    

}

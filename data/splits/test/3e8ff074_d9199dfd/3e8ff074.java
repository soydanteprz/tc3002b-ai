import java.io.*;
import java.util.*;

public class blue_red_permutation {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int T = Integer.parseInt(br.readLine());

        while (T-- > 0) {
            int N = Integer.parseInt(br.readLine());

            String line1[] = br.readLine().split(" ");
            char line2[] = br.readLine().toCharArray();

            Range ranges[] = new Range[N];
            for (int i = 0; i < N; i++) {
                int cnt = Integer.parseInt(line1[i]);
                int low = line2[i] == 'B' ? Math.min(1, cnt) : Math.max(1, cnt);
                int high = line2[i] == 'R' ? Math.max(N, cnt) : Math.min(N, cnt);
                ranges[i] = new Range(low, high);
            }

            Arrays.sort(ranges);

            boolean doable = true;
            for (int i = 0; i < N; i++) {
                if (!ranges[i].contains(i + 1)) { doable = false; break; }
            }

            System.out.println(doable ? "YES" : "NO");
        }
    }

    static class Range implements Comparable<Range> {
        final int low, high;
        
        Range(int low,int high) {
            this.low = low;
            this.high = high;
        }

        public boolean contains(int val) {
            return val >= this.low && val <= this.high;
        }

        
    }
}

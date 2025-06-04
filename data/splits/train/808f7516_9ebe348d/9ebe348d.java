import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.TreeSet;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskD solver = new TaskD();
        solver.solve(1, in, out);
        out.close();
    }

    static class TaskD {
        public void solve(int testNumber, InputReader in, PrintWriter out) {
            int N = in.nextInt();
            int[] h = new int[N];
            int[] hsort = new int[N];
            HashMap<Integer, Integer> condense = new HashMap<>();
            for (int i = 0; i < N; i++) {
                h[i] = in.nextInt();
                hsort[i] = h[i];
            }
            ArrayUtils.sort(hsort);
            int timer = 0;
            int[] t = new int[N];
            for (int i = 0; i < N; i++) {
                if (!condense.containsKey(hsort[i])) {
                    condense.put(hsort[i], timer++);
                    t[timer - 1]++;
                } else {
                    t[condense.get(hsort[i])]++;
                }
            }
            int[][] heights = new int[timer][];
            for (int i = 0; i < timer; i++) {
                heights[i] = new int[t[i]];
                t[i] = 0;
            }
            for (int i = 0; i < N; i++) {
                h[i] = condense.get(h[i]);
                heights[h[i]][t[h[i]]++] = i;
            }
            TreeSet<Integer> cur = new TreeSet<>();
            ArrayList<Integer>[] to = new ArrayList[N];
            for (int i = 0; i < N; i++) {
                to[i] = new ArrayList<>();
                if (i != N - 1) {
                    to[i].add(i + 1);
                }
            }
            int low = N + 1;
            int high = -1;
            for (int i = 0; i < timer; i++) {
                for (int e : heights[i]) {
                    cur.add(e);
                    low = Math.min(low, e);
                    high = Math.max(high, e);
                }
                for (int e : heights[i]) {
                    if (low < e) {
                        to[cur.lower(e)].add(e);
                    }
                    if (high > e) {
                        to[e].add(cur.higher(e));
                    }
                }
            }
            cur.clear();
            low = N + 1;
            high = -1;
            for (int i = timer - 1; i >= 0; i--) {
                for (int e : heights[i]) {
                    cur.add(e);
                    low = Math.min(low, e);
                    high = Math.max(high, e);
                }
                for (int e : heights[i]) {
                    if (low < e) {
                        to[cur.lower(e)].add(e);
                    }
                    if (high > e) {
                        to[e].add(cur.higher(e));
                    }
                }
            }
            long[] dp = new long[N];
            Arrays.fill(dp, (int) (1e8));
            dp[0] = 0;
            for (int i = 0; i < N; i++) {
                for (int toa : to[i]) {
                    dp[toa] = Math.min(dp[toa], dp[i] + 1);
                }
            }
            out.println(dp[N - 1]);
        }

    }

    

    static class ArrayUtils {
        public static void shuffle(int[] arr) {
            for (int i = 0; i < arr.length; i++) {
                int rand = (int) (Math.random() * (i + 1));
                swap(arr, i, rand);
            }
        }

        public static void swap(int[] arr, int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

        public static void sort(int[] arr) {
            shuffle(arr);
            Arrays.sort(arr);
            //get rid of quicksort cases
        }

    }
}

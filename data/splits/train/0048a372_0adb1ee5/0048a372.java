import java.io.*;
import java.util.*;

public class AirConditioners {
    public static long[] solve(int n, int k, HashMap<Integer, Integer> store) {
        long[] result = new long[n];
        Arrays.fill(result, Integer.MAX_VALUE);
        long current = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            current++;
            if (store.containsKey(i)) {
                current = Math.min(current, store.get(i));
            }

            result[i] = Math.min(result[i], current);
        }
        current = Integer.MAX_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            current++;
            if (store.containsKey(i)) {
                current = Math.min(current, store.get(i));
            }
            result[i] = Math.min(result[i], current);
        }
        return result;
    }

    public static void main(String[] args) {
        FastScanner fs = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int test = fs.nextInt();
        for (int t = 0; t < test; t++) {
            int n = fs.nextInt();
            int k = fs.nextInt();
            int[] array = fs.readArray(k);
            int[] temperature = fs.readArray(k);
            HashMap<Integer, Integer> store = new HashMap<>();
            for (int i = 0; i < k; i++) {
                store.put(array[i] - 1, temperature[i]);
            }
            long[] ans = solve(n, k, store);
            for (int i = 0; i < n; i++) {
                out.print(ans[i] + " ");
            }
            out.println();
        }
        out.println();
        out.close();
    }

    

    

}

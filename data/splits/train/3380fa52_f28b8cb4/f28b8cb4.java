import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Main {
    

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(System.in));

        int q = Integer.parseInt(br.readLine());

        for (int i = 0; i < q; i++) {
            br.readLine();

            StringTokenizer st = new StringTokenizer(br.readLine());
            int n = Integer.parseInt(st.nextToken());
            int k = Integer.parseInt(st.nextToken());

            // read input a
            st = new StringTokenizer(br.readLine());
            int[] a = new int[k];
            for (int j = 0; j < k; j++) {
                a[j] = Integer.parseInt(st.nextToken());
            }

            // read input t
            st = new StringTokenizer(br.readLine());
            int[] t = new int[k];
            for (int j = 0; j < k; j++) {
                t[j] = Integer.parseInt(st.nextToken());
            }

            // known temperatures
            int[] c = new int[n];
            Arrays.fill(c, Integer.MAX_VALUE);
            for (int j = 0; j < k; j++) {
                c[a[j] - 1] = t[j];
            }

            // temperature from left to right
            long[] left = new long[n];
            Arrays.fill(left, Integer.MAX_VALUE);
            long p = Integer.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                p = Math.min(p + 1, c[j]);
                left[j] = p;
            }

            // temperature from right to left
            long[] right = new long[n];
            Arrays.fill(right, Integer.MAX_VALUE);
            p = Integer.MAX_VALUE;
            for (int j = n - 1; j >= 0; j--) {
                p = Math.min(p + 1, c[j]);
                right[j] = p;
            }

            for (int j = 0; j < n; j++) {
                System.out.print(Math.min(left[j], right[j]) + " ");
            }
            System.out.println();
        }
    }
}

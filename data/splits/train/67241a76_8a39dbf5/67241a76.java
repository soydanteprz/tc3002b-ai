import java.util.*;
import java.io.*;

public class CodeForces {

    public void run() {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            int q = Integer.parseInt(br.readLine());
            while (q-- > 0){
                br.readLine();
                StringTokenizer st = new StringTokenizer(br.readLine());
                int n = Integer.parseInt(st.nextToken());
                int k = Integer.parseInt(st.nextToken());
                int[] a = new int[k];
                int[] t = new int[k];
                long[] L = new long[n];
                long[] R = new long[n];
                for (int i = 0; i < n; i++) {
                    L[i] = Integer.MAX_VALUE;
                    R[i] = Integer.MAX_VALUE;
                }
                st = new StringTokenizer(br.readLine());
                StringTokenizer st1 = new StringTokenizer(br.readLine());
                for (int i = 0; i < k; i++) {
                    a[i] = Integer.parseInt(st.nextToken());
                    t[i] = Integer.parseInt(st1.nextToken());
                    L[a[i] - 1] = t[i];
                    R[a[i] - 1] = t[i];
                }
                for (int i = 1; i < n; i++) {
                    L[i] = Math.min(L[i-1] + 1, L[i]);
                }
                for (int i = n - 2; i >= 0; i--) {
                    R[i] = Math.min(R[i], R[i + 1] + 1);
                }
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < n; i++) {
                    long tmp = Math.min(L[i], R[i]);
                    sb.append(tmp);
                    sb.append(" ");
                }
                System.out.println(sb.toString().trim());

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new CodeForces().run();
    }
}

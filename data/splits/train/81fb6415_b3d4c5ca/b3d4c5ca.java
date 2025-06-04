import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    public static long floor(long a, long b) {
        long res = a / b;
        while(res * b > a) res--;
        return res;
    }
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        while (t-- > 0) {
            StringTokenizer st = new StringTokenizer(br.readLine());
            int n = Integer.parseInt(st.nextToken());
            long k = Long.parseLong(st.nextToken());
            st = new StringTokenizer(br.readLine());
            Long[] p = new Long[n];
            for(int i = 0 ;i<n;i++) {
                p[i] = Long.parseLong(st.nextToken());
            }
            Arrays.sort(p);
            long[] sums = new long[n+1];
            for(int i=0;i<n;i++) sums[i+1] = sums[i] + p[i];
            long ans = Long.MAX_VALUE;
            for(int y=0;y<n;y++) {
                long x = p[0] - floor(k - sums[n-y] + p[0], y+1);
                ans = Math.min(Math.max(x, 0) + y, ans);
            }
            System.out.println(ans);
        }
    }
}

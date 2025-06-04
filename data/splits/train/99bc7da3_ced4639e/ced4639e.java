import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

public class Main {
    public static boolean find(int[] a) {
        int n = a.length;
        for (int i = 1; i < Math.pow(3, n); i++) {
            int sum = 0;
            int x = i;
            for (int j = 0; j < n; j++) {
                int r = x % 3;
                if (r == 1) sum += a[j];
                else if (r ==2) sum -= a[j];
                x /= 3;
            }
            if (sum == 0) return true;
        }
        return false;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter pw = new PrintWriter(System.out);
        int t = Integer.parseInt(br.readLine());
        for (int i = 0; i < t; i++) {
            int n = Integer.parseInt(br.readLine());
            int[] a = new int[n];
            String[] s = br.readLine().trim().split("\\s+");
            for (int j = 0; j < n; j++) {
                a[j] = Integer.parseInt(s[j]);
            }
            boolean res = find(a);
            if (res) pw.println("YES");
            else pw.println("NO");
        }
        pw.flush();
    }
}

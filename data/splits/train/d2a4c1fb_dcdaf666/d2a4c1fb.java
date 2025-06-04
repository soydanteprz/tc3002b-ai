import java.util.*;

public class CodeForcesRound {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        for (int i = 0; i<t; i++)
            System.out.println(solve(sc));
        sc.close();
    }

    public static long solve(Scanner sc) {
        int n = sc.nextInt();
        long[] c = new long[n];
        long[] ps = new long[n];

        c[0] = sc.nextLong();
        c[1] = sc.nextLong();
        ps[0] = c[0];
        ps[1] = c[1];

        for (int i = 2; i<n; i++) {
            c[i] = sc.nextLong();
            ps[i] = ps[i-2] + c[i];
        }

        long res = Long.MAX_VALUE;
        long[] mins = new long[n];
        mins[0] = n*c[0];
        mins[1] = n*c[1];
        int minEven = 0;
        int minOdd = 1;

        for (int i = 2; i<n; i++) {
            if (i%2==0) {
                if (c[i]<c[minEven])
                    minEven = i;

                mins[i] = ps[minEven]-c[minEven] + ps[i]-ps[minEven] + c[minEven]*(n-i/2);
            } else {
                if (c[i]<c[minOdd])
                    minOdd = i;

                mins[i] = ps[minOdd]-c[minOdd] + ps[i]-ps[minOdd] + c[minOdd]*(n-i/2);
            }
        }

        for (int i = 1; i<n; i++) {
            res = Math.min(res, mins[i]+mins[i-1]);
        }

        return res;
    }

    public static long min(long a, long b, long c) {
        long t = Math.min(a, b);
        return Math.min(t, c);
    }
}

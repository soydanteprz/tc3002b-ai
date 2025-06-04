import java.util.Scanner;

public class MinimumGridPath {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int test = scanner.nextInt();

        StringBuilder sb = new StringBuilder();
        for (int t = 0; t < test; t++) {
            int n = scanner.nextInt();
            long[] arr = new long[n];
            for (int i = 0; i < n; i++) {
                arr[i] = scanner.nextLong();
            }

            long minx = arr[0];
            long miny = arr[1];

            long min = minx * n + miny * n;
            long sumx = arr[0];
            long sumy = arr[1];

            for(int i=2; i<n; i++) {
                int xc;
                int yc;
                if(i%2 == 0) {
                    xc = i/2 + 1;
                    yc = i/2;
                    sumx += arr[i];
                    minx = Math.min(minx, arr[i]);
                } else {
                    xc = i/2 + 1;
                    yc = i/2 + 1;
                    sumy += arr[i];
                    miny = Math.min(miny, arr[i]);
                }
                min = Math.min(min, sumx + (n-xc) * minx + sumy + (n-yc) * miny);
            }


            sb.append(min).append(System.lineSeparator());
        }
        System.out.println(sb);

    }
}

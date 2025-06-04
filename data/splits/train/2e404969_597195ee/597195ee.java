import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;

public class C_MonstersAndSpells_1700 {
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        int t = sc.nextInt();
        while(t-->0) {
            int n = sc.nextInt();
            int[] times = new int[n];
            int[] health = new int[n];
            for(int i = 0; i < n; i++) {
                times[i] = sc.nextInt();
            }
            for(int i = 0; i < n; i++) {
                health[i] = sc.nextInt();
            }
            Point[] points = new Point[n];
            for(int i = 0; i < n; i++) {
                points[i] = new Point(times[i]-health[i], times[i]);
            }
            Arrays.sort(points);
            long ans = 0;
            for(int i = 0; i < n; i++) {
                int j = i+1;
                int latestTime = points[i].time;
                while(j < n && points[j].startBy < latestTime) {
                    latestTime = Math.max(latestTime, times[j]);
                    j++;
                }
                long length = latestTime-points[i].startBy;
                ans += (length*(length + 1))/2;
                i = j - 1;
            }
            System.out.println(ans);
        }

        out.close();
    }

    static class Point implements Comparable<Point> {
        Integer startBy;
        Integer time;

        Point(int startBy, int time) {
            this.startBy = startBy;
            this.time = time;
        }

        
    }

    public static PrintWriter out;

    
    
}

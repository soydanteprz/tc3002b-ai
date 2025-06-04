import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;

public class TaskC {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Solver solver = new Solver();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class Solver {
        public void solve(int testNumber, FastReader in, PrintWriter out) {
            int n = in.nextInt(),m = in.nextInt();
            int[][] a = new int[n][m];
            for(int i=0;i<n;i++){
                String s = in.next();
                for(int j=0;j<m;j++){
                    a[i][j] = (s.charAt(j)-'0');
                }
            }
            ArrayList<Pair> ans = new ArrayList<>();
            for(int i=0;i<=n-2;i++){
                for(int j=0;j<=m-2;j++) {
                    int cnt = 0;
                    if (a[i][j] == 1) cnt++;
                    if (a[i + 1][j] == 1) cnt++;
                    if(a[i][j+1]==1) cnt++;
                    if(a[i+1][j+1]==1) cnt++;
                    if(cnt==4){
                        a[i][j] = 0;
                        a[i+1][j] = 0;
                        a[i+1][j+1] = 0;
                        ans.add(new Pair((i+1),(j+1)));
                        ans.add(new Pair((i+2),(j+1)));
                        ans.add(new Pair((i+2),(j+2)));
//                        out.println();
                        cnt = 1;
                    }
                    if(cnt==1){
                        boolean flag = false;
                        for(int x=0;x<2;x++){
                            for(int y=0;y<2;y++){
                                if(a[i+x][j+y]==1){
//                                    out.print((i+1+x)+" "+(j+1+y)+" ");
                                    ans.add(new Pair((i+1+x),(j+1+y)));
                                    int cnt2 = 1;
                                    for(int sx=0;sx<2;sx++){
                                        for(int sy=0;sy<2;sy++){
                                            if(a[i+sx][j+sy]!=1){
//                                                out.print((i+sx+1)+" "+(j+1+sy)+" ");
                                                a[i+sx][j+sy]=1;
                                                ans.add(new Pair((i+sx+1),(j+sy+1)));
                                                cnt2++;
                                                if(cnt2==3){
                                                    a[i+x][j+y]=0;
                                                    flag = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if(flag) break;
                                    }
                                    if(flag) break;
                                }
                                if(flag) break;
                            }
                            if(flag) break;
                        }
//                        out.println();
                        cnt = 2;
                    }
                    if(cnt==2){
                        boolean flag = false;
                        for(int x=0;x<2;x++){
                            for(int y=0;y<2;y++){
                                if(a[i+x][j+y]==1){
//                                    out.print((i+1+x)+" "+(j+1+y)+" ");
                                    ans.add(new Pair((i+1+x),(j+1+y)));
                                    int cnt2 = 1;
                                    for(int sx=0;sx<2;sx++){
                                        for(int sy=0;sy<2;sy++){
                                            if(a[i+sx][j+sy]!=1){
//                                                out.print((i+sx+1)+" "+(j+1+sy)+" ");
                                                cnt2++;
                                                ans.add(new Pair((i+sx+1),(j+1+sy)));
                                                a[i+sx][j+sy]=1;
                                                if(cnt2==3){
                                                    a[i+x][j+y]=0;
                                                    flag = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if(flag) break;
                                    }
                                    if(flag) break;
                                }
                                if(flag) break;
                            }
                            if(flag) break;
                        }
//                        out.println();
                        cnt = 3;
                    }
                    if(cnt==3){
                        for(int x=0;x<2;x++){
                            for(int y=0;y<2;y++){
                                if(a[i+x][j+y]==1){
                                    a[i+x][j+y]=0;
                                    ans.add(new Pair((i+1+x),(j+1+y)));
//                                    out.print((i+1+x)+" "+(j+1+y)+" ");
                                }
                            }
                        }
//                            out.println();
                    }
                }
            }
            out.println((ans.size())/3);
            for(int i=0;i<ans.size();i+=3){
                for(int j=0;j<=2;j++){
                    out.print(ans.get(i+j).a+" "+ans.get(i+j).b+ " ");
                }
                out.println();
            }
        }
    }
    static class Pair{
        int a,b;
        public Pair(int a,int b){
            this.a=a;
            this.b=b;
        }
    }
    
}

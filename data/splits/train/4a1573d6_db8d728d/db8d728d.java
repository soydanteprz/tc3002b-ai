import java.util.*;
import java.io.*;
import java.math.*;
import java.lang.*;

public class BinaryTable {
    
    // static int mod = 998244353;
    static int mod = 1000000007;

    

    public static void main(String[] args) throws Exception {
        FastReader scn = new FastReader();
        PrintWriter pw = new PrintWriter(System.out);
        int t = scn.nextInt();
        outer : while(t-->0){
            int n = scn.nextInt();
            int m = scn.nextInt();
            int[][] arr = new int[n][m];
            for(int i=0; i<n; i++){
            	String s = scn.nextLine();
            	for(int j=0; j<m; j++){
            		char ch = s.charAt(j);
            		arr[i][j] = ch - '0';
            	}
            }
            ArrayList<Pair> list = new ArrayList<>();
            if(n % 2 == 0 && m % 2 == 0){
            	for(int i=0; i<n; i+=2){
            		for(int j=0; j<m; j+=2){
            			if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}
            		}
            	}
            }else if(n % 2 == 0 && m % 2 == 1){
            	int M = m-1;
            	for(int i=0; i<n; i+=2){
            		for(int j=0; j<M; j+=2){
            			if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}
            		}
            	}
            	for(int i=0; i<n; i+=2){
            		if(arr[i][m-1] == 1 && arr[i+1][m-1] == 1){
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-1));
            		}else if(arr[i][m-1] == 1){
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i, m-1));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-1));
						list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-1));
            		}else if(arr[i+1][m-1] == 1){
            			list.add(new Pair(i, m-1));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-1));
           				list.add(new Pair(i, m-2));
            		}
            	}
            }else if(n % 2 == 1 && m % 2 == 0){
            	int N = n-1;
            	for(int i=0; i<N; i+=2){
            		for(int j=0; j<m; j+=2){
            			if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}
            		}
            	}
            	for(int j=0; j<m; j+=2){
            		if(arr[n-1][j] == 1 && arr[n-1][j+1] == 1){
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-2, j+1));
            		}else if(arr[n-1][j] == 1){
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-1, j));
						list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-2, j));
            		}else if(arr[n-1][j+1] == 1){
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j));
           				list.add(new Pair(n-1, j+1));
            		}
            	}
            }else if(n % 2 == 1 && m % 2 == 1){
            	int N = n-1;
            	int M = m-1;
            	for(int i=0; i<N; i+=2){
            		for(int j=0; j<M; j+=2){
            			if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j] == 1 && arr[i][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1 && arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            			}else if(arr[i][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}else if(arr[i][j+1] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j] == 1){
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i, j));
            			}else if(arr[i+1][j+1] == 1){
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j+1));
            				list.add(new Pair(i+1, j+1));
            				list.add(new Pair(i, j));
            				list.add(new Pair(i+1, j));
            			}
            		}
            	}
            	for(int j=0; j<M; j+=2){
            		if(arr[n-1][j] == 1 && arr[n-1][j+1] == 1){
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-2, j+1));
            		}else if(arr[n-1][j] == 1){
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-1, j));
						list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-2, j));
            		}else if(arr[n-1][j+1] == 1){
            			list.add(new Pair(n-1, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j+1));
            			list.add(new Pair(n-2, j+1));
            			list.add(new Pair(n-2, j));
            			list.add(new Pair(n-1, j));
           				list.add(new Pair(n-1, j+1));
            		}
            	}
            	for(int i=0; i<N; i+=2){
            		if(arr[i][m-1] == 1 && arr[i+1][m-1] == 1){
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-1));
            		}else if(arr[i][m-1] == 1){
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i, m-1));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-1));
						list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i, m-1));
            		}else if(arr[i+1][m-1] == 1){
            			list.add(new Pair(i, m-1));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-2));
            			list.add(new Pair(i+1, m-2));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i+1, m-1));
            			list.add(new Pair(i, m-1));
           				list.add(new Pair(i, m-2));
            		}
            	}
            	if(arr[n-1][m-1] == 1){
            		list.add(new Pair(n-1, m-1));
            		list.add(new Pair(n-2, m-1));
            		list.add(new Pair(n-1, m-2));
            		list.add(new Pair(n-2, m-2));
            		list.add(new Pair(n-1, m-2));
            		list.add(new Pair(n-1, m-1));
            		list.add(new Pair(n-1, m-1));
            		list.add(new Pair(n-2, m-2));
            		list.add(new Pair(n-2, m-1));
            	}
            }
            int k = list.size()/3;
            pw.println(k);
            for(int i=0; i<k; i++){
            	for(int j=0; j<3; j++){
            		Pair p = list.get(i*3+j);
            		pw.print((p.x+1) + " " + (p.y+1) + " ");
            	}
            	pw.println();
            }
        }
        pw.close();
    }

    public static class Pair{
        int x;
        int y;

        Pair(int x, int y){
            this.x = x;
            this.y = y;
        }
    }

    

    

    // private static void sort(Pair[] arr) {
    //     List<Pair> list = new ArrayList<>();
    //     for(int i=0; i<arr.length; i++){
    //         list.add(arr[i]);
    //     }
    //     Collections.sort(list); // collections.sort uses nlogn in backend
    //     for (int i = 0; i < arr.length; i++){
    //         arr[i] = list.get(i);
    //     }
    // }

    

    

    

    

    

    

    // (x^y)%p in O(logy)
    

    
    
    
    
    

    

    // TC- O(logmax(a,b))
    

    

    

    // TC- O(logmax(a,b))
    

    

     
}
